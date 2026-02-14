import texture;

public float4 sampling_fwd(
    uint2 tid,
    uint2 canvasSize,
    StructuredBuffer<float4> sampleTex,
    uint2 texSize)
{
    // TODO: Student implementation starts here.

    // Calculate UV coordinates
    float2 uv = (float2(tid) + 0.5) / float2(canvasSize);
    DiffTexture2D<float, 4> diffTex = DiffTexture2D<float, 4>(sampleTex, texSize);
    // Map UV to texture space
    return sampleBilinear(diffTex, uv);

    // TODO: Student implementation ends here.
}

public void sampling_bwd(
    uint2 tid,
    uint2 canvasSize,
    StructuredBuffer<float4> sampleTex,
    RWStructuredBuffer<Atomic<float>[4]> sampleTexGrad,
    uint2 texSize,
    float4 dOut)
{
    // TODO: Student implementation starts here.
    // Calculate UV coordinates
    float2 uv = (float2(tid) + 0.5) / float2(canvasSize);
    DiffTexture2D_Differential<float, 4> diffTexGrad = DiffTexture2D_Differential<float, 4>(sampleTexGrad);
    DiffTexture2D<float, 4> diffTex = DiffTexture2D<float, 4>(sampleTex, texSize);
    DifferentialPtrPair<DiffTexture2D<float, 4>> diffTexPair = { diffTex, diffTexGrad };
    bwd_diff(sampleBilinear)(diffTexPair, uv, dOut);
    // TODO: Student implementation ends here.
}
