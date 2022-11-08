import { ReplayBuffer } from '../../../replay';
import { vec3, mat4 } from 'wgpu-matrix';

type VertexDesc = {
    buffer: ReplayBuffer;
    offset: number;
    stride: number;
    size: number; // 1 - 4
};

export type BufferRenderParameters = {
    position: VertexDesc;
    color?: VertexDesc;
    indexBuffer?: ReplayBuffer;
    indexBufferType?: GPUIndexFormat;
    primitiveTopology: GPUPrimitiveTopology;
    worldViewMatrix: Float32Array | number[];
    numVertices: number;
    renderColor: Float32Array | number[];
};

export class BufferRenderer {
    device: GPUDevice;
    pipelines: Map<string, GPURenderPipeline> = new Map<string, GPURenderPipeline>();
    vsUniformBuffer: GPUBuffer;
    fsUniformBuffer: GPUBuffer;
    vsUniformValues: Float32Array;
    fsUniformValues: Float32Array;
    worldViewProjection: Float32Array;
    positionDesc: Float32Array;
    colorDesc: Float32Array;
    color: Float32Array;
    vertexColorMix: Float32Array;

    constructor(device: GPUDevice) {
        this.device = device;

        const presentationFormat = navigator.gpu.getPreferredCanvasFormat(); // gpu.getPreferredCanvasFormat(adapter);

        const shaderModule = device.createShaderModule({
            code: `
    struct VSUniforms {
        worldViewProjection: mat4x4<f32>,
        worldInverseTranspose: mat4x4<f32>,
    };
    @group(0) @binding(0) var<uniform> vsUniforms: VSUniforms;

    struct MyVSInput {
            @location(0) position: vec4<f32>,
    };

    struct MyVSOutput {
        @builtin(position) position: vec4<f32>,
        @location(0) normal: vec3<f32>,
        @location(1) texcoord: vec2<f32>,
    };

    @vertex
    fn myVSMain(v: MyVSInput) -> MyVSOutput {
        var vsOut: MyVSOutput;
        vsOut.position = vsUniforms.worldViewProjection * v.position;
        return vsOut;
    }

    struct FSUniforms {
        lightDirection: vec3<f32>,
    };

    @group(0) @binding(1) var<uniform> fsUniforms: FSUniforms;
    @group(0) @binding(2) var diffuseSampler: sampler;
    @group(0) @binding(3) var diffuseTexture: texture_2d<f32>;

    @fragment
    fn myFSMain(v: MyVSOutput) -> @location(0) vec4<f32> {
        return vec4<f32>(1, 0, 0, 1) + vec4(fsUniforms.lightDirection, 0) * 0.0;
    }
       `,
        });

        function createBuffer(device: GPUDevice, data: Float32Array | Uint16Array, usage: number) {
            const buffer = device.createBuffer({
                size: data.byteLength,
                usage: usage | GPUBufferUsage.COPY_DST,
            });
            device.queue.writeBuffer(buffer, 0, data);
            return buffer;
        }

        const pos = [
            1, 1, -1, 1, 1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1,
            -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1,
            -1, 1, 1, -1, 1, -1, -1, -1, -1, -1,
        ];
        const positions = new Float32Array(
            [
                0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 8, 9, 10, 8, 10, 11, 12, 13, 14, 12, 14, 15, 16, 17, 18, 16, 18, 19,
                20, 21, 22, 20, 22, 23,
            ]
                .map(ndx => pos.slice(ndx * 3, (ndx + 1) * 3))
                .flat()
        );

        const positionBuffer = createBuffer(device, positions, GPUBufferUsage.VERTEX);

        //device.pushErrorScope('validation');
        const pipeline = device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: shaderModule,
                entryPoint: 'myVSMain',
                buffers: [
                    // position
                    {
                        arrayStride: 3 * 4, // 3 floats, 4 bytes each
                        attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }],
                    },
                ],
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'myFSMain',
                targets: [
                    {
                        format: presentationFormat,
                        blend: {
                            color: {
                                srcFactor: 'one',
                                dstFactor: 'one-minus-src-alpha',
                                operation: 'add',
                            },
                            alpha: {
                                srcFactor: 'one',
                                dstFactor: 'one-minus-src-alpha',
                                operation: 'add',
                            },
                        },
                    },
                ],
            },
            primitive: {
                topology: 'triangle-list',
                cullMode: 'none',
            },
        });

        const vUniformBufferSize = 2 * 16 * 4; // 2 mat4s * 16 floats per mat * 4 bytes per float
        const fUniformBufferSize = 3 * 4; // 1 vec3 * 3 floats per vec3 * 4 bytes per float

        const vsUniformBuffer = device.createBuffer({
            size: vUniformBufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        const fsUniformBuffer = device.createBuffer({
            size: fUniformBufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        const vsUniformValues = new Float32Array(2 * 16); // 2 mat4s
        const worldViewProjection = vsUniformValues.subarray(0, 16);
        const worldInverseTranspose = vsUniformValues.subarray(16, 32);
        const fsUniformValues = new Float32Array(3); // 1 vec3
        const lightDirection = fsUniformValues.subarray(0, 3);

        const bindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: vsUniformBuffer } },
                { binding: 1, resource: { buffer: fsUniformBuffer } },
            ],
        });

        const renderPassDescriptor = {
            colorAttachments: [
                {
                    // view: undefined, // Assigned later
                    // resolveTarget: undefined, // Assigned Later
                    clearValue: { r: 0.5, g: 0.5, b: 0.5, a: 1.0 },
                    loadOp: 'clear',
                    storeOp: 'store',
                },
            ],
            // depthStencilAttachment: {
            //     // view: undefined,    // Assigned later
            //     depthClearValue: 1,
            //     depthLoadOp: 'clear',
            //     depthStoreOp: 'store',
            // },
        };

        this.render = (context: GPUCanvasContext) => {
            const time = performance.now() * 0.001;
            const canvas = context.canvas as HTMLCanvasElement;

            const projection = mat4.perspective(
                (30 * Math.PI) / 180,
                canvas.clientWidth / canvas.clientHeight,
                0.5,
                10
            );
            const eye = [1, 4, -6];
            const target = [0, 0, 0];
            const up = [0, 1, 0];

            const camera = mat4.lookAt(eye, target, up);
            const view = mat4.inverse(camera);
            const viewProjection = mat4.multiply(projection, view);
            const world = mat4.rotationY(time);
            mat4.transpose(mat4.inverse(world), worldInverseTranspose);
            mat4.multiply(viewProjection, world, worldViewProjection);

            vec3.normalize([1, 8, -10], lightDirection);

            device.queue.writeBuffer(
                vsUniformBuffer,
                0,
                vsUniformValues.buffer,
                vsUniformValues.byteOffset,
                vsUniformValues.byteLength
            );
            device.queue.writeBuffer(
                fsUniformBuffer,
                0,
                fsUniformValues.buffer,
                fsUniformValues.byteOffset,
                fsUniformValues.byteLength
            );

            const colorTexture = context.getCurrentTexture();
            renderPassDescriptor.colorAttachments[0].view = colorTexture.createView();
            //renderPassDescriptor.depthStencilAttachment.view = canvasInfo.depthTextureView;

            const commandEncoder = device.createCommandEncoder();
            const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
            passEncoder.setPipeline(pipeline);
            passEncoder.setBindGroup(0, bindGroup);
            passEncoder.setVertexBuffer(0, positionBuffer);
            passEncoder.draw(positions.length / 3);
            passEncoder.end();
            device.queue.submit([commandEncoder.finish()]);
            requestAnimationFrame(this.render);
        };
        requestAnimationFrame(this.render);
    }

    // Get or create a texture renderer for the given device.
    static rendererCache = new WeakMap();
    static getRendererForDevice(device: GPUDevice) {
        let renderer = BufferRenderer.rendererCache.get(device);
        if (!renderer) {
            renderer = new BufferRenderer(device);
            BufferRenderer.rendererCache.set(device, renderer);
        }
        return renderer;
    }
}
