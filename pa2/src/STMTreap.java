import java.util.concurrent.atomic.AtomicLong;


public class STMTreap implements IntSet {
    static class Node {
        final int key;
        final int priority;
        Node left;
        Node right;

        Node(final int key, final int priority) {
            this.key = key;
            this.priority = priority;
        }

        @Override
		public String toString() {
            return "Node[key=" + key + ", prio=" + priority +
                    ", left=" + (left == null ? "null" : String.valueOf(left.key)) +
                    ", right=" + (right == null ? "null" : String.valueOf(right.key)) + "]";
        }
    }

    private AtomicLong randState = new AtomicLong(0L);
    private Node root;

    @Override
    @org.deuce.Atomic
	public boolean contains(final int key) {
        Node node = root;
        while (node != null) {
            if (key == node.key) {
                return true;
            }
            node = key < node.key ? node.left : node.right;
        }
        return false;
    }

    @Override
    @org.deuce.Atomic
	public void add(final int key) {
        //root = addImpl(key, randPriority());
        Node newRoot = addImpl(root, key);
        if (newRoot != root) {
            root = newRoot;
        }
    }

    private Node addImpl(final Node node, final int key) {
        if (node == null) {
            return new Node(key, randPriority());
        }
        else if (key == node.key) {
            // no insert needed
            return node;
        }
        else if (key < node.key) {
            Node newLeft = addImpl(node.left, key); 
            if (newLeft != node.left) {
                node.left = newLeft;
            }

            if (node.left.priority > node.priority) {
                return rotateRight(node);
            }
            return node;
        }
        else {
            Node newRight = addImpl(node.right, key);
            if (newRight != node.right) {
                node.right = newRight;
            }
            if (node.right.priority > node.priority) {
                return rotateLeft(node);
            }
            return node;
        }
    }

    private int randPriority() {
        // The constants in this 64-bit linear congruential random number
        // generator are from http://nuclear.llnl.gov/CNP/rng/rngman/node4.html
        long r = randState.get();
        long newR = r * 2862933555777941757L + 3037000493L;
        randState.compareAndSet(r, newR);
        return (int)(newR >> 30);
    }

    private Node rotateRight(final Node node) {
        //       node                  nL
        //     /      \             /      \
        //    nL       z     ==>   x       node
        //  /   \                         /   \
        // x   nLR                      nLR   z
        final Node nL = node.left;
        node.left = nL.right;
        nL.right = node;
        return nL;
    }

    private Node rotateLeft(final Node node) {
        final Node nR = node.right;
        node.right = nR.left;
        nR.left = node;
        return nR;
    }

    @Override
    @org.deuce.Atomic
	public void remove(final int key) {
        Node newRoot = removeImpl(root, key);
        if (newRoot != root) {
            root = newRoot;
        }
    }

    private Node removeImpl(final Node node, final int key) {
        if (node == null) {
            // not present, nothing to do
            return null;
        }
        else if (key == node.key) {
            if (node.left == null) {
                // splice out this node
                return node.right;
            }
            else if (node.right == null) {
                return node.left;
            }
            else {
                // Two children, this is the hardest case.  We will pretend
                // that node has -infinite priority, move it down, then retry
                // the removal.
                if (node.left.priority > node.right.priority) {
                    // node.left needs to end up on top
                    final Node top = rotateRight(node);
                    top.right = removeImpl(top.right, key);
                    return top;
                } else {
                    final Node top = rotateLeft(node);
                    top.left = removeImpl(top.left, key);
                    return top;
                }
            }
        }
        else if (key < node.key) {
            Node newLeft = removeImpl(node.left, key);
            if (node.left != newLeft) {
                node.left = newLeft;
            }
            return node;
        }
        else {
            Node newRight = removeImpl(node.right, key);
            if (node.right != newRight) {
                node.right = newRight;
            }
            return node;
        }
    }
}
