use std::{collections::VecDeque, fmt, ops::Deref, sync::Arc};

use ash::{ext::swapchain_maintenance1, prelude::VkResult, vk};
use parking_lot::Mutex;
use smallvec::SmallVec;

use crate::generic::{DeviceError, OutOfMemory, PipelineStages, QueueFlags};

use super::{
    device::Device, from::IntoAsh, handle_host_oom, map_device_error, map_oom, refs::Refs,
    surface::Frame, unexpected_error, CommandBuffer, CommandEncoder,
};

/// Maximum number of pending epochs to keep in queue.
/// Queue will wait for earliest epoch to be complete and reuse it
/// when number of epochs exceeds this limit.
///
/// The number is chosen to minimize waiting (ideally epoch would be already complete when it's recycled)
/// and to minimize memory usage (epoch contains resources that are not released until it's complete).
const MAX_EPOCHS: usize = 3;

/// Maximum number of command pools to keep in queue.
/// When new command buffer is needed it will allocate from the oldest used pool if it was reset.
/// Otherwise it will create a new pool if number of pools is less than this limit.
/// Otherwise it will keep using last pool
const MAX_POOLS: usize = 3;

unsafe fn deallocate_cbuf(
    cbuf: vk::CommandBuffer,
    pool: vk::CommandPool,
    pools: &mut VecDeque<Pool>,
) {
    // Safety:
    // Caller must ensure that pool exists.
    let pool = unsafe { pools.iter_mut().find(|p| p.pool == pool).unwrap_unchecked() };
    pool.deallocate(cbuf);
}

pub struct Pool {
    free_cbufs: Vec<vk::CommandBuffer>,
    pool: vk::CommandPool,
    allocated: usize,
}

impl Pool {
    fn allocate(&mut self, device: &ash::Device) -> Result<vk::CommandBuffer, OutOfMemory> {
        if let Some(cbuf) = self.free_cbufs.last() {
            unsafe {
                device.begin_command_buffer(
                    *cbuf,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
            }
            .map_err(map_oom)?;

            self.allocated += 1;
            return Ok(unsafe { self.free_cbufs.pop().unwrap_unchecked() });
        }

        let mut cbuf = vk::CommandBuffer::null();

        let result = unsafe {
            (device.fp_v1_0().allocate_command_buffers)(
                device.handle(),
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
                &mut cbuf,
            )
        };

        match result {
            vk::Result::SUCCESS => {}
            err => return Err(map_oom(err)),
        }

        let result = unsafe {
            device.begin_command_buffer(
                cbuf,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
        };

        if let Err(err) = result {
            self.free_cbufs.push(cbuf);
            return Err(map_oom(err));
        }

        self.allocated += 1;
        return Ok(cbuf);
    }

    fn deallocate(&mut self, cbuf: vk::CommandBuffer) {
        self.free_cbufs.push(cbuf);
        self.allocated -= 1;
    }
}

/// Epoch contains resource references
/// and fence that must be signaled before references can be dropped.
struct Epoch {
    fence: vk::Fence,
    refs: Vec<Refs>,

    /// Contains owning command pool handle for each command buffer in the epoch.
    cbufs: Vec<(vk::CommandBuffer, vk::CommandPool)>,
}

impl Epoch {
    /// Destroy the epoch.
    /// This is called when owning queue is dropped.
    ///
    /// # Safety
    ///
    /// Device must be the same device that created the epoch.
    /// Pools must contain all pools that were used to allocate command buffers in the epoch.
    unsafe fn destroy(&mut self, device: &ash::Device, pools: &mut VecDeque<Pool>) {
        // Safety: caller must ensure device is owner.
        unsafe {
            device.destroy_fence(self.fence, None);
        }

        for (cbuf, pool) in self.cbufs.drain(..) {
            // Safety: caller must ensure pool exists.
            unsafe {
                deallocate_cbuf(cbuf, pool, pools);
            }
        }
    }

    /// Resets the epoch for recycling.
    /// Drops all resource references and resets the fence.
    ///
    /// If this call fails the epoch is not completely reset, although resources are freed.
    /// `reset` may be called again to retry.
    ///
    /// # Safety
    ///
    /// Device must be the same device that created the epoch.
    /// Pools must contain all pools that were used to allocate command buffers in the epoch.
    unsafe fn reset(
        &mut self,
        device: &ash::Device,
        pools: &mut VecDeque<Pool>,
    ) -> Result<(), OutOfMemory> {
        self.refs.iter_mut().for_each(|r| r.clear());

        for (cbuf, pool) in self.cbufs.drain(..) {
            // Safety: caller must ensure pool exists.
            unsafe {
                deallocate_cbuf(cbuf, pool, pools);
            }
        }

        // Safety: called must ensure device is owner.
        unsafe {
            device.reset_fences(&[self.fence]).map_err(map_oom)?;
        }
        Ok(())
    }
}

struct PendingEpochs {
    array: Mutex<VecDeque<Epoch>>,
}

impl PendingEpochs {
    fn new() -> Self {
        PendingEpochs {
            array: Mutex::new(VecDeque::new()),
        }
    }

    fn push(&mut self, epoch: Epoch) {
        self.array.get_mut().push_back(epoch);
    }

    fn recycle(
        &mut self,
        device: &ash::Device,
        pools: &mut VecDeque<Pool>,
    ) -> Result<Option<Epoch>, DeviceError> {
        let mut array = self.array.get_mut();
        if array.len() < MAX_EPOCHS {
            return Ok(None);
        }

        /// Can't create new epoch, must wait for the earliest one to complete.
        unsafe {
            let front_epoch = array.front_mut().unwrap_unchecked();

            device
                .wait_for_fences(&[front_epoch.fence], true, !0)
                .map_err(map_device_error)?;
            front_epoch.reset(device, pools)?;
        }

        // Epoch is properly reset and ready to be reused.
        Ok(Some(unsafe { array.pop_front().unwrap_unchecked() }))
    }

    fn destroy_all(&mut self, device: &ash::Device, pools: &mut VecDeque<Pool>) {
        let mut array = self.array.get_mut();
        for e in array.iter_mut() {
            unsafe {
                e.destroy(device, pools);
            }
        }
    }

    /// Releases all resources but keeps the epochs.
    fn queue_is_idle(&self) {
        let mut array = self.array.lock();
        for epoch in array.iter_mut() {
            epoch.refs.clear();
        }
    }
}

pub struct Queue {
    /// Device associated with the queue.
    device: Device,

    /// Vulkan queue handle.
    handle: vk::Queue,

    /// Queue family index.
    family: u32,

    /// Queue flags.
    flags: QueueFlags,

    /// Command pools to allocate command buffers from.
    pools: VecDeque<Pool>,

    /// Free refs instances to reuse.
    /// Refs from recycled epochs are added here.
    free_refs: Vec<Refs>,

    // Waits to add into next submission
    wait_semaphores: Vec<vk::Semaphore>,

    // Stages to wait for.
    wait_stages: Vec<vk::PipelineStageFlags>,

    // Signals to add into next submission
    signal_semaphores: Vec<vk::Semaphore>,

    /// Current epoch that is being filled with resources from command buffers.
    this_epoch: Option<Epoch>,

    /// Pending epochs that are waiting for completion.
    /// Epochs might be recycled when associated fence is signaled.
    /// Or if Device::wait_idle or Queue::wait_idle wait is called.
    pending_epochs: PendingEpochs,

    /// Temporary array for command buffers.
    command_buffers: Vec<CommandBuffer>,

    /// Temporary array for command buffers to submit
    command_buffer_submit: Vec<vk::CommandBuffer>,

    // Present resources
    present_semaphores: Vec<vk::Semaphore>,
    present_swapchains: Vec<vk::SwapchainKHR>,
    present_indices: Vec<u32>,
    present_fences: Vec<vk::Fence>,
}

impl Drop for Queue {
    fn drop(&mut self) {
        let device = self.device.ash();
        unsafe {
            device.queue_wait_idle(self.handle).unwrap();

            // Queue is idle, all epochs must be complete.
            self.pending_epochs.destroy_all(device, &mut self.pools);

            if let Some(epoch) = &mut self.this_epoch {
                epoch.destroy(device, &mut self.pools);
            }

            for pool in &mut self.pools {
                debug_assert_eq!(pool.allocated, 0, "All command buffers must be deallocated");
                device.destroy_command_pool(pool.pool, None);
            }
        }
    }
}

impl fmt::Debug for Queue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Queue({:p}@{:?})", self.handle, self.device)
    }
}

impl Queue {
    pub(super) fn new(device: Device, handle: vk::Queue, flags: QueueFlags, family: u32) -> Self {
        Queue {
            device,
            handle,
            flags,
            family,
            wait_semaphores: Vec::new(),
            wait_stages: Vec::new(),
            signal_semaphores: Vec::new(),
            pools: VecDeque::new(),
            free_refs: Vec::new(),
            this_epoch: None,
            pending_epochs: PendingEpochs::new(),

            command_buffers: Vec::new(),
            command_buffer_submit: Vec::new(),
            present_semaphores: Vec::new(),
            present_swapchains: Vec::new(),
            present_indices: Vec::new(),
            present_fences: Vec::new(),
        }
    }

    pub(super) fn add_wait(&mut self, semaphores: vk::Semaphore, before: PipelineStages) {
        self.wait_semaphores.push(semaphores);
        self.wait_stages
            .push(ash::vk::PipelineStageFlags::TOP_OF_PIPE | before.into_ash());
    }

    fn refresh_pools(pools: &mut VecDeque<Pool>, device: &ash::Device) -> Result<(), OutOfMemory> {
        if let Some(front) = pools.front_mut() {
            if front.allocated == 0 {
                // If front pool has no allocated command buffers, reuse it.

                // Since pool is in array it *was* used to allocate command buffers
                // unless allocation of the first command buffer failed.
                // So don't hesitate to reset it first.

                // Keep resources allocated by the pool.

                // If resetting fails with oom, report it to the caller,
                // allocating new command buffer will probably fail too.
                unsafe {
                    device.reset_command_pool(front.pool, vk::CommandPoolResetFlags::empty())
                }
                .map_err(map_oom)?;

                /// Place the pool to the back of the queue where it will be used in `get_pool`.
                let reset_pool = unsafe { pools.pop_front().unwrap_unchecked() };
                pools.push_back(reset_pool);
            }
        }
        Ok(())
    }

    #[inline]
    fn get_pool<'a>(
        pools: &'a mut VecDeque<Pool>,
        device: &ash::Device,
    ) -> Result<&'a mut Pool, OutOfMemory> {
        let more_pools = pools.len() < MAX_POOLS;
        match pools.back() {
            Some(pool) if !more_pools || pool.allocated == 0 => {}
            _ => {
                // Create a new pool.
                // Use non-inline cold function to reduce code size.
                // As this branch would be taken only few times at the beginning of mev usage.
                #[cold]
                #[inline(never)]
                fn create_pool(
                    device: &ash::Device,
                    pools: &mut VecDeque<Pool>,
                ) -> Result<(), OutOfMemory> {
                    let pool = unsafe {
                        device.create_command_pool(
                            &vk::CommandPoolCreateInfo::default()
                                .flags(vk::CommandPoolCreateFlags::TRANSIENT),
                            None,
                        )
                    }
                    .map_err(map_oom)?;

                    let pool = Pool {
                        pool,
                        free_cbufs: Vec::new(),
                        allocated: 0,
                    };

                    pools.push_back(pool);
                    Ok(())
                }

                create_pool(device, pools)?;
            }
        }
        Ok(unsafe { pools.back_mut().unwrap_unchecked() })
    }

    /// # Safety
    ///
    /// Must be called after fence of the epoch returned by `get_epoch` is submitted.
    unsafe fn next_epoch(&mut self) {
        // Safety: caller must ensure that this_epoch is not None by calling get_epoch first.
        let epoch = unsafe { self.this_epoch.take().unwrap_unchecked() };
        self.pending_epochs.push(epoch);
    }

    /// Returns current epoch to use.
    ///
    /// If no current epoch is set:
    /// - Reuses the earliest epoch if there are more than 3 pending epochs.
    /// - Or creates a new one.
    fn get_epoch<'a>(
        this_epoch: &'a mut Option<Epoch>,
        pending_epochs: &mut PendingEpochs,
        pools: &mut VecDeque<Pool>,
        free_refs: &mut Vec<Refs>,
        device: &Device,
    ) -> Result<&'a mut Epoch, DeviceError> {
        if let Some(epoch) = this_epoch {
            return Ok(epoch);
        }

        match pending_epochs.recycle(device.ash(), pools)? {
            Some(epoch) => {
                // Always inserts since this_epoch is None.
                return Ok(this_epoch.get_or_insert(epoch));
            }
            None => {
                /// Create a new epoch fence.
                let fence = device.new_fence()?;

                // Always inserts since this_epoch is None.
                Ok(this_epoch.get_or_insert(Epoch {
                    fence,
                    refs: Vec::new(),
                    cbufs: Vec::new(),
                }))
            }
        }
    }
}

impl Deref for Queue {
    type Target = Device;

    #[inline(always)]
    fn deref(&self) -> &Device {
        &self.device
    }
}

impl crate::traits::Resource for Queue {}

#[hidden_trait::expose]
impl crate::traits::Queue for Queue {
    /// Get the device associated with this queue.
    #[inline(always)]
    fn device(&self) -> &Device {
        &self.device
    }

    /// Get the queue family index.
    #[inline(always)]
    fn family(&self) -> u32 {
        self.family
    }

    /// Create a new command encoder associated with this queue.
    /// The encoder must be submitted to the queue it was created from.
    fn new_command_encoder(&mut self) -> Result<CommandEncoder, OutOfMemory> {
        let device = self.device.ash();
        Self::refresh_pools(&mut self.pools, device)?;
        let pool = Self::get_pool(&mut self.pools, device)?;

        let device = self.device.ash();

        let handle = pool.allocate(device)?;

        Ok(CommandEncoder::new(
            self.device.clone(),
            handle,
            pool.pool,
            self.free_refs.pop().unwrap_or_else(Refs::new),
        ))
    }

    /// Submit command buffers to the queue.
    ///
    /// If `check_point` is `true`, inserts a checkpoint into queue and check previous checkpoints.
    /// Checkpoints are required for resource reclamation.
    fn submit<I>(&mut self, command_buffers: I, check_point: bool) -> Result<(), DeviceError>
    where
        I: IntoIterator<Item = CommandBuffer>,
    {
        debug_assert!(self.command_buffer_submit.is_empty());
        debug_assert!(self.command_buffers.is_empty());

        let signal_semaphores_len = self.signal_semaphores.len();
        let present_semaphores_len = self.present_semaphores.len();
        let present_swapchains_len = self.present_swapchains.len();
        let present_indices_len = self.present_indices.len();

        let epoch = match Self::get_epoch(
            &mut self.this_epoch,
            &mut self.pending_epochs,
            &mut self.pools,
            &mut self.free_refs,
            &self.device,
        ) {
            Ok(epoch) => epoch,
            Err(DeviceError::OutOfMemory) => {
                self.drop_command_buffer(command_buffers);
                return Err(DeviceError::OutOfMemory);
            }
            Err(DeviceError::DeviceLost) => return Err(DeviceError::DeviceLost),
        };

        // Add handle to list of command buffers to submit.
        // Collect frames to present and command buffers into the cache array.
        for mut cbuf in command_buffers {
            self.command_buffer_submit.push(cbuf.handle);

            for frame in &cbuf.present {
                if frame.is_real() {
                    self.signal_semaphores.push(frame.present);
                    self.present_semaphores.push(frame.present);
                    self.present_swapchains.push(frame.swapchain);
                    self.present_indices.push(frame.idx);
                    self.present_fences.push(frame.fence);
                } else {
                    self.signal_semaphores.push(frame.present);
                }
            }

            self.command_buffers.push(cbuf);
        }

        let fence = if check_point {
            epoch.fence
        } else {
            ash::vk::Fence::null()
        };

        let result = unsafe {
            self.device.ash().queue_submit(
                self.handle,
                &[vk::SubmitInfo::default()
                    .wait_semaphores(&self.wait_semaphores)
                    .wait_dst_stage_mask(&self.wait_stages)
                    .signal_semaphores(&self.signal_semaphores)
                    .command_buffers(&self.command_buffer_submit)],
                fence,
            )
        };

        self.command_buffer_submit.clear();

        match result {
            Ok(()) => {}
            Err(err) => {
                self.signal_semaphores.truncate(signal_semaphores_len);
                self.present_semaphores.truncate(present_semaphores_len);
                self.present_swapchains.truncate(present_swapchains_len);
                self.present_indices.truncate(present_indices_len);

                match err {
                    vk::Result::ERROR_OUT_OF_HOST_MEMORY => {
                        self.command_buffers.clear();
                        handle_host_oom()
                    }
                    vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => {
                        // Attempt to reclaim some resources.
                        for mut cbuf in self.command_buffers.drain(..) {
                            cbuf.refs.clear();
                            self.free_refs.push(cbuf.refs);

                            unsafe {
                                deallocate_cbuf(cbuf.handle, cbuf.pool, &mut self.pools);
                            }
                        }
                        return Err(DeviceError::OutOfMemory);
                    }
                    vk::Result::ERROR_DEVICE_LOST => {
                        // Nothing can be done now.
                        self.command_buffers.clear();
                        return Err(DeviceError::DeviceLost);
                    }
                    _ => unexpected_error(err),
                }
            }
        }

        // Drain refs from command buffers and add them to the epoch
        // when submitting was successful.
        for cbuf in self.command_buffers.drain(..) {
            epoch.refs.push(cbuf.refs);
            epoch.cbufs.push((cbuf.handle, cbuf.pool));
        }

        self.wait_semaphores.clear();
        self.wait_stages.clear();
        self.signal_semaphores.clear();

        if check_point {
            unsafe {
                self.next_epoch();
            }
        }

        if !self.present_swapchains.is_empty() {
            debug_assert_eq!(self.present_swapchains.len(), self.present_indices.len());
            debug_assert_eq!(self.present_swapchains.len(), self.present_semaphores.len());
            debug_assert_eq!(self.present_swapchains.len(), self.present_fences.len());

            let mut present_info = vk::PresentInfoKHR::default()
                .swapchains(&self.present_swapchains)
                .wait_semaphores(&self.present_semaphores)
                .image_indices(&self.present_indices);

            let mut present_fence = vk::SwapchainPresentFenceInfoEXT::default();
            if let Some(swapchain_maintenance1) = self.device.swapchain_maintenance1() {
                present_fence = present_fence.fences(&self.present_fences);
                present_info = present_info.push_next(&mut present_fence);
            }

            let result = unsafe {
                self.device
                    .swapchain()
                    .queue_present(self.handle, &present_info)
            };

            match result {
                Ok(_) => {
                    self.present_semaphores.clear();
                    self.present_swapchains.clear();
                    self.present_indices.clear();
                    self.present_fences.clear();
                }
                Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY) => handle_host_oom(),
                Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY) => {
                    return Err(DeviceError::OutOfMemory)
                }
                Err(vk::Result::ERROR_DEVICE_LOST) => return Err(DeviceError::DeviceLost),
                Err(
                    vk::Result::ERROR_OUT_OF_DATE_KHR
                    | vk::Result::ERROR_SURFACE_LOST_KHR
                    | vk::Result::ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT,
                ) => {
                    // Images are released and semaphores are queued.
                    self.present_semaphores.clear();
                    self.present_swapchains.clear();
                    self.present_indices.clear();
                    self.present_fences.clear();
                }
                Err(err) => unexpected_error(err),
            };
        }
        Ok(())
    }

    /// Drop command buffers without submitting them to the queue.
    fn drop_command_buffer<I>(&mut self, command_buffers: I)
    where
        I: IntoIterator<Item = CommandBuffer>,
    {
        for mut cbuf in command_buffers {
            cbuf.refs.clear();
            self.free_refs.push(cbuf.refs);

            unsafe {
                deallocate_cbuf(cbuf.handle, cbuf.pool, &mut self.pools);
            }
        }
    }

    /// Synchronize the access to the frame resources.
    fn sync_frame(&mut self, frame: &mut Frame, before: PipelineStages) {
        assert!(!frame.synced, "Frame must be synced exactly once");

        if frame.acquire != vk::Semaphore::null() {
            self.add_wait(frame.acquire, before);
        }

        frame.synced = true;
    }

    fn wait_idle(&self) -> Result<(), OutOfMemory> {
        let result = unsafe { self.device.ash().queue_wait_idle(self.handle) };

        let result = result.map_err(|err| match err {
            ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY => handle_host_oom(),
            ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => OutOfMemory,
            ash::vk::Result::ERROR_DEVICE_LOST => unimplemented!("Device lost"),
            _ => unexpected_error(err),
        });

        self.pending_epochs.queue_is_idle();

        result
    }
}
