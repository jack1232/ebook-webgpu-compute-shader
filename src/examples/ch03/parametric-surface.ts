import vsShader from '../../common/shader-vert.wgsl';
import fsShader from '../../common/shader-frag.wgsl';
import csColormap from '../../common/colormap.wgsl';
import csSurfaceFunc from './parametric-surface-func.wgsl';
import csSurface from './parametric-surface-comp.wgsl';
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
        //vertexBuffers: [positionBuffer, normalBuffer, colorBuffer, colorBuffer2, indexBuffer, indexBuffer2],  
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
    const csShader = csColormap.concat(csSurfaceFunc.concat(csSurface));
    const descriptor = ws.createComputePipelineDescriptor(device, csShader);
    const csPipeline = await device.createComputePipelineAsync(descriptor);

    const vertexBuffer = ws.createBuffer(device, numVertices * vertexByteSize, ws.BufferType.VertexStorage);

    const csParamsBufferSize = 
        1 * 4 + // resolution: u32
        1 * 4 + // funcSelection: u32
        1 * 4 + // colormapSelection: u32
        1 * 4 + // wireframeColormapSelection: u32
        1 * 4 + // colormapDirection: u32
        1 * 4 + // colormapReverse: u32
        2 * 4 + // padding: u32
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

const draw = (init:ws.IWebGPUInit, p:ws.IPipeline, p2:ws.IPipeline, p3: ws.IPipeline, plotType: string) => {  
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
    const deviceDescriptor: GPUDeviceDescriptor = {
        requiredLimits:{
            maxStorageBufferBindingSize: 512*1024*1024 //512MiB, defaulting to 128MiB
        }
    }
    const init = await ws.initWebGPU({canvas, msaaCount: 4}, deviceDescriptor);

    let p = await createPipeline(init);
    let p2 = await createComputePipeline(init.device);
    let p3 = await createComputeIndexPipeline(init.device);
    
    var gui =  ws.getDatGui();
    const params = {
        rotationSpeed: 1,
        surfaceType: 'klein bottle',
        plotType: 'surface',                     
        resolution: 64,
        colormap: 'jet',
        wireframeColor: 'white', 
        colormapDirection: 'y',
        colormapReverse: false,
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
    let funcSelection = 10;
    let resolotionChanged = false;
           
    gui.add(params, 'surfaceType', [
            'astroid', 'astroid 2', 'astroidal torus', 'bohemian dome', 'boy shape', 'breather', 'enneper', 'figure 8', 'henneberg', 'kiss',
            'klein bottle', 'klein bottle 2', 'klein bottle 3', 'kuen', 'minimal', 'parabolic cyclide', 'pear', 'plucker conoid',
            'seashell', 'sievert-enneper', 'steiner', 'torus', 'wellenkugel'
        ]).onChange((val:string) => {
            if(val === 'astroid') funcSelection = 0;
            else if(val === 'astroid 2') funcSelection = 1;
            else if(val === 'astroidal torus') funcSelection = 2;
            else if(val === 'bohemian dome') funcSelection = 3;
            else if(val === 'boy shape') funcSelection = 4;
            else if(val === 'breather') funcSelection = 5;
            else if(val === 'enneper') funcSelection = 6;
            else if(val === 'figure 8') funcSelection = 7;
            else if(val === 'henneberg') funcSelection = 8;
            else if(val === 'kiss') funcSelection = 9;
            else if(val === 'klein bottle') funcSelection = 10;
            else if(val === 'klein bottle 2') funcSelection = 11;
            else if(val === 'klein bottle 3') funcSelection = 12;
            else if(val === 'kuen') funcSelection = 13;
            else if(val === 'minimal') funcSelection = 14;
            else if(val === 'parabolic cyclide') funcSelection = 15;
            else if(val === 'pear') funcSelection = 16;
            else if(val === 'plucker conoid') funcSelection = 17;
            else if(val === 'seashell') funcSelection = 18;
            else if(val === 'sievert-enneper') funcSelection = 19;
            else if(val === 'steiner') funcSelection = 20;
            else if(val === 'torus') funcSelection = 21;
            else if(val === 'wellenkugel') funcSelection = 22;
    });
    gui.add(params, 'rotationSpeed', 0, 5, 0.1);
   
    var folder = gui.addFolder('Set Surface Parameters');
    folder.open();
    folder.add(params, 'plotType', ['surface', 'wireframe', 'surface_wireframe']);
    folder.add(params, 'resolution', 16, 2048, 16).onChange(() => {      
        resolotionChanged = true;
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
        
        if(resolotionChanged){
            resolution = params.resolution;
            numVertices = resolution * resolution;
            numTriangles = 6 * (resolution - 1) * (resolution - 1);
            numLines = 8 * (resolution - 1) * (resolution - 1);
            p2 = await createComputePipeline(init.device);
            p3 = await createComputeIndexPipeline(init.device);
            resolotionChanged = false;
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
        init.device.queue.writeBuffer(p2.uniformBuffers[0], 0, Uint32Array.of(
            resolution,
            funcSelection,
            colormapSelection,
            wirefremeColor,
            colormapDirection,
            colormapReverse,
            0,  // padding
            0            
        ));
        
        draw(init, p, p2, p3, params.plotType);      
        requestAnimationFrame(frame);
        stats.end();
    };
    frame();
}

run();