const pi:f32 = 3.14159265359;

struct VertexData{
    position: vec3f,
    normal: vec3f,
    color: vec3f,
    color2: vec3f,
}

struct VertexDataArray{
    vertexDataArray: array<VertexData>,
}

struct SuperShapeParams {
    n1:vec4f,
    n2:vec4f,
    a1:vec2f,
    a2:vec2f,
    resolution: f32,
    colormapSelection: f32,
    wireframeColormapSelection: f32,
    colormapDirection: f32,
    colormapReverse: f32,
    animationTime: f32,
    scaling:f32,
    aspectRatio: f32,   
}

@group(0) @binding(0) var<storage, read_write> vda : VertexDataArray;
@group(0) @binding(1) var<uniform> ssp: SuperShapeParams;

var<private> umin:f32; 
var<private> umax:f32; 
var<private> vmin:f32; 
var<private> vmax:f32;

fn superShape3D(u:f32, v:f32, t:f32, n1:vec4f, n2:vec4f, a1:vec2f, a2:vec2f) -> vec3f {
    var raux1 = pow(abs(1.0 / a1.x * cos(n1.x * u /4.0)), n1.z) + pow(abs(1.0 / a1.y * sin(n1.x * u /4.0)), n1.w);
    var r1 = pow(abs(raux1), -1.0 / n1.y);
    var raux2 = pow(abs(1.0 / a2.x * cos(n2.x * v /4.0)), n2.z) + pow(abs(1.0 / a2.y * sin(n2.x * v /4.0)), n2.w);
    var r2 = pow(abs(raux2), -1.0 / n2.y);

    var a = 0.334*(2.0 + sin(t)); 
    var v1 = v*a;
    var x = r1 * cos(u) * r2 * cos(v1);
    var y = r2 * sin(v1);
    var z = r1 * sin(u) * r2 * cos(v1);
    return vec3(x, y, z);
}

fn getUv(id: vec3u) -> vec2f {
    umin = -pi;
    umax = pi;
    vmin = -0.5*pi;
    vmax = 0.5*pi;

    var du = (umax - umin)/(f32(ssp.resolution) - 1.0);
    var dv = (vmax - vmin)/(f32(ssp.resolution) - 1.0);
    var u = umin + f32(id.x) * du;
    var v = vmin + f32(id.y) * dv;
    return vec2(u, v);
}

fn normalizePoint(u:f32, v:f32) -> vec3f {
    var t = ssp.animationTime;
    var pos = superShape3D(u, v, t, ssp.n1, ssp.n2, ssp.a1, ssp.a2);
    pos.y = pos.y*ssp.aspectRatio;
    return pos*ssp.scaling;
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) id : vec3u) {
    var i = id.x;
    var j = id.y;
    var idx = i + j * u32(ssp.resolution);
    var uv = getUv(id);
    
    var p0 = normalizePoint(uv.x, uv.y);

    // calculate normals
    var p1:vec3f;
    var p2:vec3f; 
    var p3:vec3f;
    let eps = 0.0001;

    if (uv.x - eps >= 0.0) {
        p1 = normalizePoint(uv.x - eps, uv.y);
        p2 = p0 - p1;
    }
    else {
        p1 = normalizePoint(uv.x + eps, uv.y);
        p2 = p1 - p0;
    }
    if (uv.y - eps >= 0.0) {
        p1 = normalizePoint(uv.x, uv.y - eps);
        p3 = p0 - p1;
    }
    else {
        p1 = normalizePoint(uv.x, uv.y + eps);
        p3 = p1 - p0;
    }
    var normal = normalize(cross(p2, p3));

    // colormap
    var range = 1.0;
    if(ssp.colormapDirection == 1){
        range = ssp.aspectRatio;
    }
var color = colorLerp(u32(ssp.colormapSelection), -range, range, p0[u32(ssp.colormapDirection)],
    u32(ssp.colormapReverse));
var color2 = colorLerp(u32(ssp.wireframeColormapSelection), -range, range, 
    p0[u32(ssp.colormapDirection)], u32(ssp.colormapReverse));

    vda.vertexDataArray[idx].position = p0;
    vda.vertexDataArray[idx].normal = normal;
    vda.vertexDataArray[idx].color = color;
    vda.vertexDataArray[idx].color2 = color2;
}