Programming Assignment #2
John Doherty
doherty1

The great thing about STM is that not many changes are required to parallelize a fairly complex data structure like a treap. In this case all that was required was to make add(), remove(), and contains() atomic and slightly modify them to remove potential conflicts. Since contains() only performs reads, no changes were required. For add() and remove() I added checks to make sure we were only writing to nodes or root if the structure of the tree had actually changed. For example in cases where we are setting node.left = newNode, I would add a check to make sure newNode was actually, in fact different from the original value of node.left. This removes extra transactional writes without changing the functionality of the data structure. 

In addition to removing conflicts by preventing unnecessary writes, I escaped the generation of the random state by storing randState as AtomicLong. This means conflicts on the generation of a randState do not cause the entire transaction to fail. To ensure that we still generate just one random value for a particular state, I made the update of the state atomic using a busy wait loop and compareAndSet().

In the end the STM system seemed to scale well and on was able to achieve better performance than the CoarseLockTreap at 8 threads on AWS. It also passes all of the correctness checks.
