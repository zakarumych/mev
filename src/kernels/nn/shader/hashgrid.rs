use std::fmt::Write as _;

use super::super::{HashGrid, HashGridLevel, hashgrid::HashGridLevelLayout};

pub(super) fn functions(index: usize, grid: &HashGrid, auxiliary: &str, trainable: bool) -> String {
    let mut s = String::new();
    let width = grid.output_arity();
    writeln!(s, "fn nn_hashgrid_{index}_forward(input:array<Scalar,3>)->array<Scalar,{width}>{{var result:array<Scalar,{width}>;").unwrap();
    for (level, layout) in grid.levels().iter().zip(grid.level_layouts()) {
        level_code(&mut s, level, layout, auxiliary, false);
    }
    s.push_str("return result;}\n");
    writeln!(
        s,
        "fn nn_hashgrid_{index}_backward(input:array<Scalar,3>,grad:array<Scalar,{width}>){{"
    )
    .unwrap();
    if trainable {
        for (level, layout) in grid.levels().iter().zip(grid.level_layouts()) {
            level_code(&mut s, level, layout, auxiliary, true);
        }
    }
    s.push_str("}\n");
    s
}

fn level_code(
    s: &mut String,
    level: &HashGridLevel,
    layout: &HashGridLevelLayout,
    auxiliary: &str,
    backward: bool,
) {
    let resolution = level.resolution;
    let features = level.features;
    let output_offset = layout.output_offset;
    writeln!(s, "{{let position=clamp(vec3<Scalar>(input[0],input[1],input[2]),vec3<Scalar>(0),vec3<Scalar>(1))*Scalar({resolution});").unwrap();
    s.push_str("let integral=floor(position);let base=vec3<u32>(integral);let fraction=position-integral;\nvar indices:array<vec3<u32>,4>;var weights:array<vec3<Scalar>,4>;\nfor(var corner=0u;corner<4u;corner++){let bit_a=corner&1u;let bit_b=corner>>1u;\n");
    writeln!(s, "let a=min(base+vec3<u32>(bit_a),vec3<u32>({resolution}u));let b=min(base+vec3<u32>(bit_b),vec3<u32>({resolution}u));").unwrap();
    s.push_str("indices[corner]=vec3<u32>(");
    for (plane, (a, b)) in [("a.x", "b.y"), ("a.y", "b.z"), ("a.z", "b.x")]
        .into_iter()
        .enumerate()
    {
        if plane != 0 {
            s.push(',');
        }
        let plane_offset = layout.parameter_offset + plane as u32 * layout.entries * features;
        if layout.dense {
            write!(
                s,
                "{plane_offset}u+({a}+{b}*{}u)*{features}u",
                resolution + 1
            )
            .unwrap();
        } else {
            write!(
                s,
                "{plane_offset}u+(({a}^({b}*2654435761u))%{}u)*{features}u",
                layout.entries
            )
            .unwrap();
        }
    }
    s.push_str(");\nlet weight_a=select(vec3<Scalar>(1)-fraction,fraction,bit_a!=0u);let weight_b=select(vec3<Scalar>(1)-fraction,fraction,bit_b!=0u);\nweights[corner]=vec3<Scalar>(weight_a.x*weight_b.y,weight_a.y*weight_b.z,weight_a.z*weight_b.x);}\n");
    writeln!(s, "for(var feature=0u;feature<{features}u;feature++){{var planes=vec3<Scalar>(0);for(var corner=0u;corner<4u;corner++){{let entry=indices[corner]+vec3<u32>(feature);planes+=weights[corner]*vec3<Scalar>(Scalar({auxiliary}[entry.x]),Scalar({auxiliary}[entry.y]),Scalar({auxiliary}[entry.z]));}}").unwrap();
    if backward {
        writeln!(s, "let delta=grad[{output_offset}u+feature];let plane_grad=vec3<Scalar>(delta*planes.y*planes.z,delta*planes.x*planes.z,delta*planes.x*planes.y);").unwrap();
        writeln!(s, "for(var corner=0u;corner<4u;corner++){{let entry=indices[corner]+vec3<u32>(feature);let value=weights[corner]*plane_grad;nn_add_{auxiliary}_gradient(entry.x,value.x);nn_add_{auxiliary}_gradient(entry.y,value.y);nn_add_{auxiliary}_gradient(entry.z,value.z);}}").unwrap();
    } else {
        writeln!(
            s,
            "result[{output_offset}u+feature]=planes.x*planes.y*planes.z;"
        )
        .unwrap();
    }
    s.push_str("}}\n");
}
