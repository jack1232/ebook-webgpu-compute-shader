struct VertexData{
    position: vec3f,
    normal: vec3f,
    color: vec3f,
    color2: vec3f,
}

struct VertexDataArray{
    vertexDataArray: array<VertexData>,
}

struct SimpleSurfaceParams {
    resolution: f32,
    funcSelection: f32,
    colormapSelection: f32,
    wireframeColormapSelection: f32,
    colormapDirection: f32,
    colormapReverse: f32,
    animationTime: f32,
    aspectRatio: f32,
}

@group(0) @binding(0) var<storage, read_write> vda : VertexDataArray;
@group(0) @binding(1) var<uniform> ssp: SimpleSurfaceParams;

var<private> xmin:f32;
var<private> xmax:f32;
var<private> ymin:f32; 
var<private> ymax:f32;
var<private> zmin:f32; 
var<private> zmax:f32;
var<private> aspect:f32;

fn getUv(i:u32, j:u32) -> vec2f {
    var dr = getDataRange(u32(ssp.funcSelection));
	xmin = dr.xRange[0];
	xmax = dr.xRange[1];
	ymin = dr.yRange[0];
	ymax = dr.yRange[1];
	zmin = dr.zRange[0];
	zmax = dr.zRange[1];	

    var dx = (xmax - xmin)/(ssp.resolution - 1.0);
    var dz = (zmax - zmin)/(ssp.resolution - 1.0);
    var x = xmin + f32(i) * dx;
    var z = zmin + f32(j) * dz;
    return vec2(x, z);
}

fn normalizePoint(u:f32, v:f32) -> vec3f {
    var pos = simpleSurfaceFunc(u, v, ssp.animationTime, u32(ssp.funcSelection));
    pos.x = 2.0 * (pos.x - xmin)/(xmax - xmin) - 1.0;
    pos.y = 2.0 * (pos.y - ymin)/(ymax - ymin) - 1.0;
    pos.z = 2.0 * (pos.z - zmin)/(zmax - zmin) - 1.0;
    pos.y = pos.y * ssp.aspectRatio;
    return pos;
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) id : vec3u) {
    let i = id.x;
    let j = id.y;   
    var uv = getUv(i, j);
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
    if(u32(ssp.colormapDirection) == 1u){
        range = ssp.aspectRatio;
    }
    let color = colorLerp(u32(ssp.colormapSelection), -range, range, p0[u32(ssp.colormapDirection)], 
        u32(ssp.colormapReverse));
    let color2 = colorLerp(u32(ssp.wireframeColormapSelection), -range, range, 
        p0[u32(ssp.colormapDirection)], u32(ssp.colormapReverse));
   
    var idx = i + j * u32(ssp.resolution);
    vda.vertexDataArray[idx].position = p0;
    vda.vertexDataArray[idx].normal = normal;
    vda.vertexDataArray[idx].color = color;
    vda.vertexDataArray[idx].color2 = color2;
}