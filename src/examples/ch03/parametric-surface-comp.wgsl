struct VertexData{
    position: vec3f,
    normal: vec3f,
    color: vec3f,
    color2: vec3f,
}

struct VertexDataArray{
    vertexDataArray: array<VertexData>, 
}

struct ParametricSurfaceParams{
    resolution: u32,
    funcSelection: u32,
    colormapSelection: u32,
    wireframeColormapSelection: u32,
    colormapDirection: u32,
    colormapReverse: u32,
}

@group(0) @binding(0) var<storage, read_write> vda : VertexDataArray;
@group(0) @binding(1) var<uniform> psp: ParametricSurfaceParams;

var<private> umin:f32; 
var<private> umax:f32; 
var<private> vmin:f32; 
var<private> vmax:f32;
var<private> xmin:f32;
var<private> xmax:f32;
var<private> ymin:f32; 
var<private> ymax:f32;
var<private> zmin:f32; 
var<private> zmax:f32;
var<private> range:f32 = 1.0;

fn getUv(id: vec3u) -> vec2f {
    var dr = getDataRange(psp.funcSelection);
    umin = dr.uRange[0];
	umax = dr.uRange[1];
	vmin = dr.vRange[0];
	vmax = dr.vRange[1];
	xmin = dr.xRange[0];
	xmax = dr.xRange[1];
	ymin = dr.yRange[0];
	ymax = dr.yRange[1];
	zmin = dr.zRange[0];
	zmax = dr.zRange[1];	

    var du = (umax - umin)/(f32(psp.resolution) - 1.0);
    var dv = (vmax - vmin)/(f32(psp.resolution) - 1.0);
    var u = umin + f32(id.x) * du;
    var v = vmin + f32(id.y) * dv;
    return vec2(u, v);
}

fn normalizePoint(u:f32, v:f32) -> vec3f {
    var pos = parametricSurfaceFunc(u, v, psp.funcSelection);
    var distance = max(max(xmax - xmin, ymax - ymin), zmax - zmin);

    if(psp.colormapDirection == 0u){
        range = (xmax - xmin)/distance;
    } else if(psp.colormapDirection == 2){
        range = (zmax - zmin)/distance;
    } else {
        range = (ymax - ymin)/distance;
    }

    pos.x = 2.0 * (pos.x - xmin)/(xmax - xmin) - 1.0;
    pos.y = 2.0 * (pos.y - ymin)/(ymax - ymin) - 1.0;
    pos.z = 2.0 * (pos.z - zmin)/(zmax - zmin) - 1.0;

    pos.x = pos.x * (xmax - xmin)/distance;
    pos.y = pos.y * (ymax - ymin)/distance;
    pos.z = pos.z * (zmax - zmin)/distance;
    
    return pos;
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) id : vec3u) {
    var i = id.x;
    var j = id.y;
    var idx = i + j * psp.resolution;
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
    var color = colorLerp(psp.colormapSelection, -range, range, p0[psp.colormapDirection], psp.colormapReverse);
    var color2 = colorLerp(psp.wireframeColormapSelection, -range, range, p0[psp.colormapDirection], psp.colormapReverse);

    vda.vertexDataArray[idx].position = p0;
    vda.vertexDataArray[idx].normal = normal;
    vda.vertexDataArray[idx].color = color; 
    vda.vertexDataArray[idx].color2 = color2; 
}