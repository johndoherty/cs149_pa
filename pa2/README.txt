Programming Assignment #2
John Doherty
doherty1

The great thing about STM is that not many changes are required to parallelize a fairly complex data structure like a treap. In this case all that was required was to make add(), remove(), and contains() atomic and slightly modify them to remove potential conflicts. Since contains() only performs reads, no changes were required. For add() and remove() I added checks to make sure we were only writing to nodes or root if the structure of the tree had actually changed. For example in cases where we are setting node.left = newNode, I would add a check to make sure newNode was actually, in fact different from the original value of node.left. This removes extra transactional writes without changing the functionality of the data structure.
