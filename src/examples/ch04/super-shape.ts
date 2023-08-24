import vsShader from '../../common/shader-vert.wgsl';
import fsShader from '../../common/shader-frag.wgsl';
import csColormap from '../../common/colormap.wgsl';
import csSurface from './super-shape-comp.wgsl';
import csIndices from '../../common/indices-comp.wgsl';
import * as ws from 'webgpu-simplified';
import { colormapDict, getIdByColormapName } from '../../common/colormap-selection';
import { vec3, mat4 } from 'gl-matrix';

var resolution = 64;
var numVertices = resolution * resolution;
var numTriangles = 6 * (resolution - 1) * (resolution - 1);
var numLines = 8 * (resolution - 1) * (resolution - 1);

const positionOffset = 0;
const normalOffset = 4 * 4;
const colorOffset = 2 * 4 * 4;
const colorOffset2 = 3 * 4 * 4;
const vertexByteSize = 
    3 * 4 + // position: vec3f
    1 * 4 + // padding f32
    3 * 4 + // normal: vec3f
    1 * 4 + // padding f32
    3 * 4 + // color: vec3f
    1 * 4 + // padding: f32
    3 * 4 + // color2: vec3f
    1 * 4 + // padding: f32
    0;

const createPipeline = async (init: ws.IWebGPUInit): Promise<ws.IPipeline> => {
    // pipeline for shape
    const descriptor = ws.createRenderPipelineDescriptor({
        init, vsShader, fsShader,
        buffers: ws.setVertexBuffers(['float32x3', 'float32x3', 'float32x3'], //pos, norm, col 
            [positionOffset, normalOffset, colorOffset], vertexByteSize),
    })
    const pipeline = await init.device.createRenderPipelineAsync(descriptor);

    // pipeline for wireframe
    const descriptor2 = ws.createRenderPipelineDescriptor({
        init, vsShader, fsShader,
        primitiveType: 'line-list',
        buffers: ws.setVertexBuffers(['float32x3', 'float32x3', 'float32x3'], //pos, norm, col2 
            [positionOffset, normalOffset, colorOffset2], vertexByteSize),
    })
    const pipeline2 = await init.device.createRenderPipelineAsync(descriptor2);
   
    // uniform buffer for transform matrix
    const  vertUniformBuffer = ws.createBuffer(init.device, 192);
    
    // uniform buffer for light 
    const lightUniformBuffer = ws.createBuffer(init.device, 48);
   
    // uniform buffer for material
    const materialUniformBuffer = ws.createBuffer(init.device, 16);

    // uniform bind group for vertex shader
    const vertBindGroup = ws.createBindGroup(init.device, pipeline.getBindGroupLayout(0), [vertUniformBuffer]);
    const vertBindGroup2 = ws.createBindGroup(init.device, pipeline2.getBindGroupLayout(0), [vertUniformBuffer]);
   
    // uniform bind group for fragment shader
    const fragBindGroup = ws.createBindGroup(init.device, pipeline.getBindGroupLayout(1), 
        [lightUniformBuffer, materialUniformBuffer]);
    const fragBindGroup2 = ws.createBindGroup(init.device, pipeline2.getBindGroupLayout(1), 
        [lightUniformBuffer, materialUniformBuffer]);

   // create depth view
   const depthTexture = ws.createDepthTexture(init);

   // create texture view for MASS (count = 4)
   const msaaTexture = ws.createMultiSampleTexture(init);

    return {
        pipelines: [pipeline, pipeline2], 
        uniformBuffers: [
            vertUniformBuffer,    // for vertex
            lightUniformBuffer,   // for fragment
            materialUniformBuffer      
        ],
        uniformBindGroups: [vertBindGroup, fragBindGroup, vertBindGroup2, fragBindGroup2],
        depthTextures: [depthTexture],
        gpuTextures: [msaaTexture],
    };
}

const createComputeIndexPipeline = async (device: GPUDevice): Promise<ws.IPipeline> => {   
    const descriptor = ws.createComputePipelineDescriptor(device, csIndices);
    const csIndexPipeline = await device.createComputePipelineAsync(descriptor);

    const indexBuffer = ws.createBuffer(device, numTriangles * 4, ws.BufferType.IndexStorage);
    const indexBuffer2 = ws.createBuffer(device, numLines * 4, ws.BufferType.IndexStorage);
    const indexUniformBuffer = ws.createBuffer(device, 4);

    device.queue.writeBuffer(indexUniformBuffer, 0, Uint32Array.of(resolution));
    const indexBindGroup = ws.createBindGroup(device, csIndexPipeline.getBindGroupLayout(0), 
        [indexBuffer, indexBuffer2, indexUniformBuffer]); 

    const indexEncoder = device.createCommandEncoder();
    const indexPass = indexEncoder.beginComputePass();
    indexPass.setPipeline(csIndexPipeline);
    indexPass.setBindGroup(0, indexBindGroup);
    indexPass.dispatchWorkgroups(Math.ceil(resolution / 8), Math.ceil(resolution / 8));
    indexPass.end();
    device.queue.submit([indexEncoder.finish()]);

    return {
        vertexBuffers:[indexBuffer, indexBuffer2]
    };
}

const createComputePipeline = async (device:GPUDevice): Promise<ws.IPipeline> => {
    const csShader = csColormap.concat(csSurface);
    const descriptor = ws.createComputePipelineDescriptor(device, csShader);
    const csPipeline = await device.createComputePipelineAsync(descriptor);

    const vertexBuffer = ws.createBuffer(device, numVertices * vertexByteSize, ws.BufferType.VertexStorage);

    const csParamsBufferSize = 
        4 * 4 + // n1: vec4f
        4 * 4 + // n2: vec4f
        2 * 4 + // a1: vec2f
        2 * 4 + // a2: vec2f
        1 * 4 + // resolution: f32
        1 * 4 + // colormapSelection: f32
        1 * 4 + // wireframeColormapSelection: f32
        1 * 4 + // colormapDirection: f32
        1 * 4 + // colormapReverse: f32
        1 * 4 + // animationTime: f32
        1 * 4 + // scaling: f32
        1 * 4 + // aspectRatio: f32
        0;

    const csParamsBuffer = ws.createBuffer(device, csParamsBufferSize);
    const csBindGroup = ws.createBindGroup(device, csPipeline.getBindGroupLayout(0), [vertexBuffer, csParamsBuffer]);

    return {
        csPipelines: [csPipeline],
        vertexBuffers: [vertexBuffer],
        uniformBuffers: [csParamsBuffer],
        uniformBindGroups: [csBindGroup],        
    }
}

const draw = (init:ws.IWebGPUInit, p:ws.IPipeline, p2:ws.IPipeline, p3:ws.IPipeline, plotType: string) => {  
    const commandEncoder =  init.device.createCommandEncoder();    
    
    // compute pass
    {
        const csPass = commandEncoder.beginComputePass();
        csPass.setPipeline(p2.csPipelines[0]);
        csPass.setBindGroup(0, p2.uniformBindGroups[0]);
        csPass.dispatchWorkgroups(Math.ceil(resolution / 8), Math.ceil(resolution / 8));
        csPass.end();
    }
    
    // render pass
    {
        const descriptor = ws.createRenderPassDescriptor({
            init,
            depthView: p.depthTextures[0].createView(),
            textureView: p.gpuTextures[0].createView(),
        });
        const renderPass = commandEncoder.beginRenderPass(descriptor);
        
        // draw surface
        function drawSurface() {
            renderPass.setPipeline(p.pipelines[0]);
            renderPass.setVertexBuffer(0, p2.vertexBuffers[0]);
            renderPass.setBindGroup(0, p.uniformBindGroups[0]);
            renderPass.setBindGroup(1, p.uniformBindGroups[1]);
            renderPass.setIndexBuffer(p3.vertexBuffers[0], 'uint32');
            renderPass.drawIndexed(numTriangles);
        }

        // draw wireframe
        function drawWireframe(){
            renderPass.setPipeline(p.pipelines[1]);
            renderPass.setVertexBuffer(0, p2.vertexBuffers[0]);
            renderPass.setBindGroup(0, p.uniformBindGroups[2]);
            renderPass.setBindGroup(1, p.uniformBindGroups[3]);
            renderPass.setIndexBuffer(p3.vertexBuffers[1], 'uint32');
            renderPass.drawIndexed(numLines);
        }

        if(plotType === 'surface'){
            drawSurface();
        } else if(plotType === 'wireframe'){
            drawWireframe();
        } else {
            drawSurface();
            drawWireframe();
        }
        renderPass.end();
    }
    init.device.queue.submit([commandEncoder.finish()]);
}

const run = async () => {
    const canvas = document.getElementById('canvas-webgpu') as HTMLCanvasElement;
    const init = await ws.initWebGPU({canvas, msaaCount: 4});

    let p = await createPipeline(init);
    let p2 = await createComputePipeline(init.device);
    let p3 = await createComputeIndexPipeline(init.device);
    
    var gui =  ws.getDatGui();
    const params = {
        rotationSpeed: 1,
        animateSpeed: 1,
        plotType: 'surface',                     
        resolution: 64,
        colormap: 'jet',
        wireframeColor: 'white', 
        colormapDirection: 'y',
        colormapReverse: false,
        param1: '7,0.2,1.7,1.7,1,1',
        param2: '7,0.2,1.7,1.7,1,1',
        scale: 1,
        aspectRatio: 0.8,
        specularColor: '#aaaaaa',
        ambient: 0.1,
        diffuse: 0.7,
        specular: 0.4,
        shininess: 30,
    };
    
    let colormapSelection = 0;
    let wirefremeColor = 13;
    let colormapDirection = 1;
    let colormapReverse = 0;
    let n1 = [7,0.2,1.7,1.7];
    let n2 = [7,0.2,1.7,1.7];
    let a1 = [1,1];
    let a2 = [1,1];
    let resolutionChanged = false;
    
    gui.add(params, 'param1').onChange((val:string) => {
        let v = val.split(',').map((e) =>Number(e));
        n1 = v.slice(0,4);
        a1 = v.slice(-2);
    });
    gui.add(params, 'param2').onChange((val:string) => {
        let v = val.split(',').map((e) =>Number(e));
        n2 = v.slice(0,4);
        a2 = v.slice(-2);
    });
    gui.add(params, 'animateSpeed', 0, 5, 0.1);   
    gui.add(params, 'rotationSpeed', 0, 5, 0.1);
   
    var folder = gui.addFolder('Set Surface Parameters');
    folder.open();
    folder.add(params, 'plotType', ['surface', 'wireframe', 'surface_wireframe']);
    folder.add(params, 'scale', 0.1, 5, 0.1); 
    folder.add(params, 'aspectRatio', 0.1, 1, 0.1); 
    folder.add(params, 'resolution', 16, 1664, 16).onChange(() => {      
        resolutionChanged = true;
    });
    
    folder.add(params, 'colormap', [
        'autumn', 'black', 'blue', 'bone', 'cool', 'cooper', 'cyan', 'fuchsia', 'green', 'greys', 'hsv', 'hot', 
        'jet', 'rainbow', 'rainbow_soft', 'red', 'spring', 'summer', 'white', 'winter', 'yellow'
    ]).onChange((val:string) => {
        colormapSelection = getIdByColormapName(colormapDict, val);
    }); 
    folder.add(params, 'colormapDirection', [
        'x', 'y', 'z'
    ]).onChange((val:string) => {              
        if(val === 'x') colormapDirection = 0;
        else if(val === 'z') colormapDirection = 2;
        else colormapDirection = 1;
    }); 
    folder.add(params, 'colormapReverse').onChange((val:boolean) => {
        if(val) colormapReverse = 1;
        else colormapReverse = 0;
    });
    folder.add(params, 'wireframeColor', [
        'black', 'blue', 'cyan', 'fuchsia', 'green', 'red', 'white', 'yellow', 'autumn', 'bone', 'cool', 
        'copper', 'greys', 'hsv', 'hot', 'jet', 'rainbow', 'rainbow_soft', 'spring', 'summer', 'winter',
    ]).onChange((val:string) => {
        wirefremeColor = getIdByColormapName(colormapDict, val);
    });

    folder = gui.addFolder('Set Lighting Parameters');
    folder.open();
    folder.add(params, 'ambient', 0, 1, 0.02);  
    folder.add(params, 'diffuse', 0, 1, 0.02);  
    folder.addColor(params, 'specularColor');
    folder.add(params, 'specular', 0, 1, 0.02);  
    folder.add(params, 'shininess', 0, 300, 1);  

    let modelMat = mat4.create();
    let normalMat = mat4.create();
    let vt = ws.createViewTransform([1.5, 1.5, 1.5]);
    let viewMat = vt.viewMat;

    let aspect = init.size.width / init.size.height;  
    let rotation = vec3.fromValues(0, 0, 0);  
    let projectMat = ws.createProjectionMat(aspect);  
    let vpMat = ws.combineVpMat(viewMat, projectMat);
   
    var camera = ws.getCamera(canvas, vt.cameraOptions);
    let eyePosition = new Float32Array(vt.cameraOptions.eye);
    let lightDirection = new Float32Array([-0.5, -0.5, -0.5]);
    init.device.queue.writeBuffer(p.uniformBuffers[0], 0, vpMat as ArrayBuffer);
    init.device.queue.writeBuffer(p.uniformBuffers[1], 0, lightDirection);
    init.device.queue.writeBuffer(p.uniformBuffers[1], 16, eyePosition);

    let start = performance.now();
    let stats = ws.getStats();

    const frame = async () => {     
        stats.begin();

        if(resolutionChanged){
            resolution = params.resolution;
            numVertices = resolution * resolution;
            numTriangles = 6 * (resolution - 1) * (resolution - 1);
            numLines = 8 * (resolution - 1) * (resolution - 1);
            p2 = await createComputePipeline(init.device);
            p3 = await createComputeIndexPipeline(init.device);
            resolutionChanged = false;
        }

        projectMat = ws.createProjectionMat(aspect); 
        if(camera.tick()){
            viewMat = camera.matrix;
            vpMat = ws.combineVpMat(viewMat, projectMat);
            eyePosition = new Float32Array(camera.eye.flat());
            init.device.queue.writeBuffer(p.uniformBuffers[0], 0, vpMat as ArrayBuffer);
            init.device.queue.writeBuffer(p.uniformBuffers[1], 16, eyePosition);
        }
        var dt = (performance.now() - start)/1000;   
        rotation[0] = Math.sin(dt * params.rotationSpeed);
        rotation[1] = Math.cos(dt * params.rotationSpeed); 
        modelMat = ws.createModelMat([0,0,0], rotation);
        normalMat = ws.createNormalMat(modelMat);
        
        // update uniform buffers for transformation 
        init.device.queue.writeBuffer(p.uniformBuffers[0], 64, modelMat as ArrayBuffer);  
        init.device.queue.writeBuffer(p.uniformBuffers[0], 128, normalMat as ArrayBuffer);  
       
        // update uniform buffers for specular light color
        init.device.queue.writeBuffer(p.uniformBuffers[1], 32, ws.hex2rgb(params.specularColor));
       
        // update uniform buffer for material
        init.device.queue.writeBuffer(p.uniformBuffers[2], 0, new Float32Array([
            params.ambient, params.diffuse, params.specular, params.shininess
        ]));
        
        // update uniform buffer for compute shader
        init.device.queue.writeBuffer(p2.uniformBuffers[0], 0, Float32Array.of(
            n1[0], n1[1], n1[2], n1[3],
            n2[0], n2[1], n2[2], n2[3],
            a1[0], a1[1],
            a2[0], a2[1],
            resolution,
            colormapSelection,
            wirefremeColor,
            colormapDirection,
            colormapReverse,
            params.animateSpeed * dt,
            params.scale,
            params.aspectRatio,            
        ));
        
        draw(init, p, p2, p3, params.plotType);      
    
        requestAnimationFrame(frame);
        stats.end();
    };
    frame();
}

run();