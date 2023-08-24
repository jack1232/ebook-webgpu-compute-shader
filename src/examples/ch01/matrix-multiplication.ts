import * as ws from 'webgpu-simplified';
import csShader from './matrix-multiplication.wgsl';

var resultBufferSize:number;
const createComputePipeline = async (device:GPUDevice): Promise<ws.IPipeline> => {
    const descriptor = ws.createComputePipelineDescriptor(device, csShader);
    const csPipeline = await device.createComputePipelineAsync(descriptor);

    const firstMatrix = new Float32Array([
        2 /* rows */, 4 /* columns */,
        1, 2, 3, 4,
        5, 6, 7, 8
    ]);
    const firstMatrixBuffer = ws.createBufferWithData(device, firstMatrix, ws.BufferType.Storage);

    const secondMatrix = new Float32Array([
        4 /* rows */, 2 /* columns */,
        1, 2,
        3, 4,
        5, 6,
        7, 8
    ]);
    const secondMatrixBuffer = ws.createBufferWithData(device, secondMatrix, ws.BufferType.Storage);

    resultBufferSize = Float32Array.BYTES_PER_ELEMENT * (2 + firstMatrix[0] * secondMatrix[1]);
    const resultMatrixBuffer = ws.createBuffer(device, resultBufferSize, ws.BufferType.Storage);
    const readBuffer = device.createBuffer({
        size: resultBufferSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    const csBindGroup = ws.createBindGroup(device, csPipeline.getBindGroupLayout(0), [
        firstMatrixBuffer, secondMatrixBuffer, resultMatrixBuffer
    ]);
    
    return {
        csPipelines: [csPipeline],
        uniformBuffers: [firstMatrixBuffer, secondMatrixBuffer, resultMatrixBuffer, readBuffer],
        uniformBindGroups: [csBindGroup],        
    }
}

const run = async () => {
    const canvas = document.getElementById('canvas-webgpu') as HTMLCanvasElement;
    const init = await ws.initWebGPU({canvas});

    let p = await createComputePipeline(init.device);

    const commandEncoder = init.device.createCommandEncoder();
    const csPass = commandEncoder.beginComputePass();
    csPass.setPipeline(p.csPipelines[0]);
    csPass.setBindGroup(0, p.uniformBindGroups[0]);
    csPass.dispatchWorkgroups(2, 4, 1);
    csPass.end();

    // Encode commands for copying buffer to buffer.
    commandEncoder.copyBufferToBuffer(
        p.uniformBuffers[2] /* source buffer */,
        0 /* source offset */,
        p.uniformBuffers[3] /* destination buffer */,
        0 /* destination offset */,
        resultBufferSize /* size */
    );

    init.device.queue.submit([commandEncoder.finish()]);

    await p.uniformBuffers[3].mapAsync(GPUMapMode.READ);
    const arrayBuffer = p.uniformBuffers[3].getMappedRange();
    console.log(new Float32Array(arrayBuffer));
}

 run();