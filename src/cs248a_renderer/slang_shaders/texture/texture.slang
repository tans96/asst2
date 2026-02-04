implementing "../texture.slang";

public struct SharedTexture2DBuffer<T>
    where T : IFloat
{
    public StructuredBuffer<T> buffer;

    T getBufferValue(SharedTexture2D tex, uint2 texelCoord)
    {
        uint2 texSize = tex.size;
        uint offset = tex.offset;
        uint index = texelCoord.y * texSize.x + texelCoord.x;
        return buffer[offset + index];
    }

    public T pointSample(SharedTexture2D tex, float2 uv)
    {
        T outV = T(0.0);
        // TODO: Student implementation starts here.

        // TODO: Student implementation ends here.
        return outV;
    }

    public T bilinearSample(SharedTexture2D tex, float2 uv)
    {
        T outV = T(0.0);
        // TODO: Student implementation starts here.

        // TODO: Student implementation ends here.
        return outV;
    }

    public T trilinearSample(SharedTexture2D[] mipmapLevels, float2 uv, float d, uint totalLevels)
    {
        T outV = T(0.0);
        // TODO: Student implementation starts here.

        // TODO: Student implementation ends here.
        return outV;
    }
}

public struct SharedTexture2D
{
    public uint2 size;
    public uint offset;
}

public struct SharedTexture3DBuffer<T>
    where T : IFloat
{
    public StructuredBuffer<T> buffer;

    T getBufferValue(SharedTexture3D tex, uint3 voxelCoord)
    {
        uint3 texSize = tex.size;
        uint offset = tex.offset;
        uint index = voxelCoord.z * texSize.y * texSize.x +
                     voxelCoord.y * texSize.x +
                     voxelCoord.x;
        return buffer[offset + index];
    }

    /**
     * Sample the texture at the given uvw coordinates using nearest neighbor interpolation.
     * @param tex The texture to sample.
     * @param uvw The uvw coordinates to sample the texture at.
     * @return The sampled value.
     */
    public T pointSample(SharedTexture3D tex, float3 uvw)
    {
        // TODO: Student implementation starts here.
        uint neearestX = min(uint(uvw.x * float(tex.size.x) + 0.5), tex.size.x - 1);
        uint neearestY = min(uint(uvw.y * float(tex.size.y) + 0.5), tex.size.y - 1);
        uint neearestZ = min(uint(uvw.z * float(tex.size.z) + 0.5), tex.size.z - 1);

        return getBufferValue(tex, uint3(neearestX, neearestY, neearestZ));

        // TODO: Student implementation ends here.
    }

    /**
     * Sample the texture at the given uvw coordinates using trilinear interpolation.
     * @param tex The texture to sample.
     * @param uvw The uvw coordinates to sample the texture at.
     * @return The sampled value.
     **/
    public T trilinearSample(SharedTexture3D tex, float3 uvw)
    {
        // TODO: Student implementation starts here.
        float floatx = uvw.x * float(tex.size.x);
        float floaty = uvw.y * float(tex.size.y);
        float floatz = uvw.z * float(tex.size.z);
        uint bottomX = min(uint(floatx), tex.size.x - 1);
        uint bottomY = min(uint(floaty), tex.size.y - 1);
        uint bottomZ = min(uint(floatz), tex.size.z - 1);
        float3 weight = float3(floatx - bottomX,
                               floaty - bottomY,
                               floatz - bottomZ);
        T result = T(0.0);
        for (uint dz = 0; dz <= 1; dz++)
            for (uint dy = 0; dy <= 1; dy++)
                for (uint dx = 0; dx <= 1; dx++)
                {
                    uint x = min(bottomX + dx, tex.size.x - 1);
                    uint y = min(bottomY + dy, tex.size.y - 1);
                    uint z = min(bottomZ + dz, tex.size.z - 1);
                    T c = getBufferValue(tex, uint3(x, y, z));

                    float wx = dx == 0 ? (1.0 - weight.x) : weight.x;
                    float wy = dy == 0 ? (1.0 - weight.y) : weight.y;
                    float wz = dz == 0 ? (1.0 - weight.z) : weight.z;
                    result = result + c * T(wx * wy * wz);
                }

        return result;

        // TODO: Student implementation ends here.
    }
}

public struct SharedTexture3D
{
    public uint3 size;
    public uint offset;
}

