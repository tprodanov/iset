
use super::*;

impl<T: PartialOrd + Copy, V, Ix: IndexType> IntervalMap<T, V, Ix> {
    /// Swap values and intervals for `i`-th and `j`-th nodes.
    unsafe fn swap_nodes(&mut self, i: Ix, j: Ix) {
        let ptr = self.nodes.as_mut_ptr();
        let ptr_i = ptr.add(i.get());
        let ptr_j = ptr.add(j.get());
        (*ptr_i).swap_with(&mut *ptr_j);
    }

    fn fix_intervals_up(&mut self, mut ix: Ix) {
        while ix.defined() {
            self.update_subtree_interval(ix);
            ix = self.nodes[ix.get()].parent;
        }
    }

    /// Removes node at index i by swapping it with the last node.
    /// This function updates all links that lead to the node that was previously the last node.
    fn swap_remove(&mut self, ix: Ix) -> V {
        let i = ix.get();
        let removed_val = self.nodes.swap_remove(i).value;
        if i >= self.nodes.len() {
            // Removed node was the last, no swap was made.
            return removed_val;
        }

        let ix = Ix::new(i).unwrap();
        let left = self.nodes[i].left;
        if left.defined() {
            self.nodes[left.get()].parent = ix;
        }

        let right = self.nodes[i].right;
        if right.defined() {
            self.nodes[right.get()].parent = ix;
        }

        let parent = self.nodes[i].parent;
        let old_ix = Ix::new(self.nodes.len()).unwrap();
        if parent.defined() {
            let parent_node = &mut self.nodes[parent.get()];
            if parent_node.left == old_ix {
                parent_node.left = ix;
            } else {
                debug_assert!(parent_node.right == old_ix);
                parent_node.right = ix;
            }
        }
        removed_val
    }

    #[inline]
    fn node_is_black(&self, ix: Ix) -> bool {
        !ix.defined() || self.nodes[ix.get()].is_black()
    }

    fn remove_child(&mut self, parent: Ix, child: Ix) {
        let parent_node = &mut self.nodes[parent.get()];
        if parent_node.left == child {
            parent_node.left = Ix::MAX;
        } else {
            debug_assert!(parent_node.right == child);
            parent_node.right = Ix::MAX;
        }
    }

    fn set_child(&mut self, parent: Ix, child: Ix, left_side: bool) {
        if child.defined() {
            self.nodes[child.get()].parent = parent;
        }
        if left_side {
            self.nodes[parent.get()].left = child;
        } else {
            self.nodes[parent.get()].right = child;
        }
    }

    fn replace_children(&mut self, prev_child: Ix, new_child: Ix) {
        let parent = self.nodes[prev_child.get()].parent;
        if parent.defined() {
            if self.nodes[parent.get()].left == prev_child {
                self.nodes[parent.get()].left = new_child;
            } else {
                self.nodes[parent.get()].right = new_child;
            }
            self.nodes[new_child.get()].parent = parent;
        } else {
            self.nodes[new_child.get()].parent = Ix::MAX;
            self.root = new_child;
        }
    }

    /// Restructure the tree before removing ix.
    /// It is known that the node is black and has black children.
    fn restructure_rm_complex_cases(&mut self, mut ix: Ix) {
        loop {
            let node = &self.nodes[ix.get()];
            debug_assert!(node.is_black());
            let parent_ix = node.parent;

            // Case (terminal): Node is the root of the tree.
            if !parent_ix.defined() {
                debug_assert!(node.is_black());
                debug_assert!(self.root == ix);
                return;
            }

            let parent = &self.nodes[parent_ix.get()];
            let parent_black = parent.is_black();
            let node_is_left = parent.left == ix;
            let sibling_ix = if node_is_left { parent.right } else { parent.left };
            let (close_nephew_ix, distant_nephew_ix) = if sibling_ix.defined() {
                let sibling = &self.nodes[sibling_ix.get()];
                if node_is_left { (sibling.left, sibling.right) } else { (sibling.right, sibling.left) }
            } else {
                (Ix::MAX, Ix::MAX)
            };

            let sibling_black = self.node_is_black(sibling_ix);
            let close_nephew_black = self.node_is_black(close_nephew_ix);
            let distant_nephew_black = self.node_is_black(distant_nephew_ix);

            if parent_black && close_nephew_black && distant_nephew_black {
                if sibling_black {
                    // Case: Node has black parent and black sibling, both nephews are black.
                    self.nodes[sibling_ix.get()].set_red();
                    ix = parent_ix;
                } else {
                    // Case: Node has black parent and red sibling, both nephews are black.
                    self.nodes[parent_ix.get()].set_red();
                    self.nodes[sibling_ix.get()].set_black();
                    self.replace_children(parent_ix, sibling_ix);
                    self.set_child(sibling_ix, parent_ix, node_is_left);
                    self.set_child(parent_ix, close_nephew_ix, !node_is_left);
                }
            }
            // Case (terminal): Node has red parent, sibling and both nephews are black.
            else if !parent_black && sibling_black && close_nephew_black && distant_nephew_black {
                self.nodes[parent_ix.get()].set_black();
                self.nodes[sibling_ix.get()].set_red();
                return;
            }
            // Case: Node has black parent, sibling and distant nephew, but close nephew is red.
            else if parent_black && sibling_black && distant_nephew_black && !close_nephew_black {
                self.nodes[close_nephew_ix.get()].set_black();
                self.nodes[sibling_ix.get()].set_red();
                self.set_child(sibling_ix, Ix::MAX, node_is_left);
                self.set_child(close_nephew_ix, sibling_ix, !node_is_left);
                self.set_child(parent_ix, close_nephew_ix, !node_is_left);
                self.update_subtree_interval(sibling_ix);
                self.update_subtree_interval(close_nephew_ix);
            }
            // Case (terminal): any parent, black sibling, any close sibling and any red distant nephew.
            else {
                debug_assert!(sibling_black && !distant_nephew_black);
                // parent's color -> subling's color.
                self.nodes[sibling_ix.get()].red_color = self.nodes[parent_ix.get()].red_color;
                self.nodes[parent_ix.get()].set_black();
                self.nodes[distant_nephew_ix.get()].set_black();
                self.replace_children(parent_ix, sibling_ix);
                self.set_child(parent_ix, close_nephew_ix, !node_is_left);
                self.set_child(sibling_ix, parent_ix, node_is_left);
                return;
            }
        }
    }

    /// Restructure the tree before removing `ix`.
    fn restructure_rm(&mut self, ix: Ix, child_ix: Ix) {
        if self.nodes[ix.get()].is_red() {
            // Both of the children must be NIL.
            debug_assert!(!child_ix.defined());
            // Do nothing.
        } else if !self.node_is_black(child_ix) {
            self.nodes[child_ix.get()].set_red();
            // Child will be removed later.
        } else {
            self.restructure_rm_complex_cases(ix);
        }
    }

    fn remove_at(&mut self, ix: Ix) -> Option<V> {
        if !ix.defined() {
            return None;
        }

        let node = &self.nodes[ix.get()];
        let rm_ix = if !node.right.defined() || !node.right.defined() {
            ix
        } else {
            // Searching for a minimal node in the right subtree.
            let mut curr = node.right;
            loop {
                let left = self.nodes[curr.get()].left;
                if !left.defined() {
                    break curr;
                }
                curr = left;
            }
        };
        if rm_ix != ix {
            unsafe { self.swap_nodes(ix, rm_ix); }
        }

        let rm_node = &self.nodes[rm_ix.get()];
        let child_ix = core::cmp::min(rm_node.left, rm_node.right);
        self.restructure_rm(rm_ix, child_ix);

        if child_ix.defined() {
            // Removed node has a child, replace the node with the child and remove the child.
            unsafe { self.swap_nodes(rm_ix, child_ix); }
            self.remove_child(rm_ix, child_ix);
            self.fix_intervals_up(rm_ix);
            Some(self.swap_remove(child_ix))
        } else {
            // Removed node has no child, just remove the node.
            let parent_ix = self.nodes[rm_ix.get()].parent;
            if parent_ix.defined() {
                self.remove_child(parent_ix, rm_ix);
                self.fix_intervals_up(parent_ix);
            } else {
                debug_assert!(self.len() == 1 && self.root == rm_ix);
                self.root = Ix::MAX;
            }
            Some(self.swap_remove(rm_ix))
        }
    }

    /// Removes an entry, associated with `interval` (exact match is required), takes $O(log n)$.
    /// Returns value if the interval was present in the map, and None otherwise.
    /// If several intervals match the query interval, removes any one of them (order is unspecified).
    pub fn remove(&mut self, interval: Range<T>) -> Option<V> {
        self.remove_at(self.find_index(&interval))
    }
}
