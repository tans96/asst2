implementing "../model.slang";
import math;

public struct BVHNode
{
    public BoundingBox bound;
    public int left;
    public int right;
    public int primLeft;
    public int primRight;
    public int depth;

    public __init(BoundingBox bound, int left = -1, int right = -1, int primLeft = -1, int primRight = -1, int depth = -1)
    {
        this.bound = bound;
        this.left = left;
        this.right = right;
        this.primLeft = primLeft;
        this.primRight = primRight;
        this.depth = depth;
    }

    public float2 hit(Ray ray)
    {
        return this.bound.hit(ray);
    }
}

public struct BVH<T>
    where T : IRayCastable
{
    public struct HitResult
    {
        public T primitive;
        public RayHitResult rayHitResult;
    }

    // Slang buffer of BVHNode copied from Python.
    public RWStructuredBuffer<BVHNode> nodes;
    // Maximum number of nodes in the BVH, in this case, it'll be the size of the nodes buffer.
    public uint maxNodes;
    // Slang buffer of primitives represented by the BVH.
    public RWStructuredBuffer<T> primitives;
    // Number of primitives in the primitives buffer.
    public uint numPrimitives;

    public __init(RWStructuredBuffer<BVHNode> nodes, RWStructuredBuffer<T> primitives, uint numPrimitives, uint maxNodes)
    {
        this.nodes = nodes;
        this.primitives = primitives;
        this.numPrimitives = numPrimitives;
        this.maxNodes = maxNodes;
    }

    struct HitStack<int N>
    {
        uint[N] nodeIndices;
        uint top;

        __init()
        {
            this.top = 0;
        }

        [mutating]
        void push(uint nodeIndex)
        {
            this.nodeIndices[this.top] = nodeIndex;
            this.top += 1;
        }

        [mutating]
        uint pop()
        {
            this.top -= 1;
            return this.nodeIndices[this.top];
        }

        bool isEmpty()
        {
            return this.top == 0;
        }

        uint size()
        {
            return this.top;
        }
    }

    // BVH traversal to find the closest hit.
    public Optional<HitResult> hit(Ray ray)
    {
        // Initialize traversal.
        Optional<HitResult> hitResult = Optional<HitResult>();
        HitStack<64> stack;

        Ray clippedRay = ray;
        // TODO: Student implementation starts here.
        stack.push(0); // Start with root node
        float closestT = float.maxValue;
        while (!stack.isEmpty())
        {
            uint nodeIndex = stack.pop();
            BVHNode node = this.nodes[nodeIndex];
            float2 tHit = node.hit(clippedRay);
            if (tHit.x > tHit.y || tHit.y < clippedRay.tRange.x || tHit.x > clippedRay.tRange.y || tHit.x > closestT)
            {
                continue; // No intersection with this node or farther than closest hit
            }
            if (node.left == -1 && node.right == -1)
            {
                // Leaf node, test all primitives
                for (int i = node.primLeft; i < node.primRight; i++)
                {
                    Optional<RayHitResult> primitiveHit = this.primitives[i].hit(clippedRay);
                    if (primitiveHit.hasValue && primitiveHit.value.t < closestT)
                    {
                        closestT = primitiveHit.value.t;
                        clippedRay.tRange.y = closestT;
                        HitResult result;
                        result.primitive = this.primitives[i];
                        result.rayHitResult = primitiveHit.value;
                        hitResult = Optional<HitResult>(result);
                    }
                }
            }
            else
            {
                // Internal node, push children to stack, but only if they exist and prioritize closer ones
                float2 leftHit = node.left == -1 ? float2(float.maxValue, float.maxValue) : this.nodes[node.left].hit(clippedRay);
                float2 rightHit = node.right == -1 ? float2(float.maxValue, float.maxValue) : this.nodes[node.right].hit(clippedRay);
                if (leftHit.x <= rightHit.x)
                {
                    if (node.right != -1 && rightHit.x <= closestT)
                    {
                        stack.push(node.right);
                    }
                    if (node.left != -1 && leftHit.x <= closestT)
                    {
                        stack.push(node.left);
                    }
                }
                else
                {
                    if (node.left != -1 && leftHit.x <= closestT)
                    {
                        stack.push(node.left);
                    }
                    if (node.right != -1 && rightHit.x <= closestT)
                    {
                        stack.push(node.right);
                    }
                }
            }
        }

        // TODO: Student implementation ends here.

        return hitResult;
    }
}
