use super::*;

impl<T: PartialOrd + Copy, V, Ix: IndexType> IntervalMap<T, V, Ix> {
    /// Swap values and intervals for `i`-th and `j`-th nodes.
    fn swap_nodes(&mut self, i: Ix, j: Ix) {
        let ptr_i = core::ptr::addr_of_mut!(self.nodes[i.get()].value);
        let ptr_j = core::ptr::addr_of_mut!(self.nodes[j.get()].value);
        unsafe {
            core::ptr::swap(ptr_i, ptr_j);
        }

        let tmp = self.node(i).interval.clone();
        self.node_mut(i).interval = self.node(j).interval.clone();
        self.node_mut(j).interval = tmp;

        let tmp = self.node(i).subtree_interval.clone();
        self.node_mut(i).subtree_interval = self.node(j).subtree_interval.clone();
        self.node_mut(j).subtree_interval = tmp;
    }

    /// Removes node at index i by swapping it with the last node.
    /// This function updates all links that lead to the node that was previously the last node.
    fn swap_remove(&mut self, ix: Ix) -> V {
        let i = ix.get();
        self.colors.swap_remove(i);
        let removed_val = self.nodes.swap_remove(i).value;
        if i >= self.nodes.len() {
            // Removed node was the last, no swap was made.
            return removed_val;
        }

        let ix = Ix::new(i).unwrap();
        if let Some(left) = self.node(ix).left {
            self.node_mut(left).parent = Some(ix);
        }
        if let Some(right) = self.node(ix).right {
            self.node_mut(right).parent = Some(ix);
        }

        let old_ix = Ix::new(self.nodes.len()).unwrap();
        if let Some(parent) = self.node(ix).parent {
            let parent_node = self.node_mut(parent);
            if parent_node.left == Some(old_ix) {
                parent_node.left = Some(ix);
            } else {
                debug_assert!(parent_node.right == Some(old_ix));
                parent_node.right = Some(ix);
            }
        }

        if self.root == Some(old_ix) {
            self.root = Some(ix);
        }
        removed_val
    }

    fn remove_child(&mut self, parent: Ix, child: Ix) {
        let parent_node = self.node_mut(parent);
        if parent_node.left == Some(child) {
            parent_node.left = None;
        } else {
            debug_assert!(parent_node.right == Some(child));
            parent_node.right = None;
        }
    }

    fn set_child(&mut self, parent: Ix, child: Option<Ix>, left_side: bool) {
        if let Some(child) = child {
            self.node_mut(child).parent = Some(parent);
        }
        if left_side {
            self.node_mut(parent).left = child;
        } else {
            self.node_mut(parent).right = child;
        }
    }

    fn replace_children(&mut self, prev_child: Ix, new_child: Ix) {
        match self.node(prev_child).parent {
            Some(parent) => {
                let parent_node = self.node_mut(parent);
                if parent_node.left == Some(prev_child) {
                    parent_node.left = Some(new_child);
                } else {
                    parent_node.right = Some(new_child);
                }
                self.node_mut(new_child).parent = Some(parent);
            }
            None => {
                self.node_mut(new_child).parent = None;
                self.root = Some(new_child);
            }
        }
    }

    /// Restructure the tree before removing ix.
    /// It is known that the node is black and has black children.
    fn restructure_rm_complex_cases(&mut self, mut ix: Ix) {
        loop {
            debug_assert!(self.is_black(ix));
            let node = self.node(ix);
            let parent_ix = match node.parent {
                Some(parent) => parent,
                None => {
                    // Case (terminal): Node is the root of the tree.
                    debug_assert!(self.root == Some(ix));
                    return;
                }
            };

            let parent = self.node(parent_ix);
            let parent_black = self.is_black(parent_ix);
            let node_is_left = parent.left == Some(ix);
            let sibling_ix = if node_is_left {
                parent.right
            } else {
                parent.left
            };
            let (close_nephew_ix, distant_nephew_ix) = if let Some(sibling_ix) = sibling_ix {
                let sibling = self.node(sibling_ix);
                if node_is_left {
                    (sibling.left, sibling.right)
                } else {
                    (sibling.right, sibling.left)
                }
            } else {
                (None, None)
            };

            let sibling_black = self.is_black_or_nil(sibling_ix);
            let close_nephew_black = self.is_black_or_nil(close_nephew_ix);
            let distant_nephew_black = self.is_black_or_nil(distant_nephew_ix);

            // XXX not clear why this is OK, we just handled it being None above
            let sibling_ix = sibling_ix.unwrap();

            if parent_black && close_nephew_black && distant_nephew_black {
                if sibling_black {
                    // Case: Node has black parent and black sibling, both nephews are black.
                    self.set_red(sibling_ix);
                    ix = parent_ix;
                } else {
                    // Case: Node has black parent and red sibling, both nephews are black.
                    self.set_red(parent_ix);
                    self.set_black(sibling_ix);
                    self.replace_children(parent_ix, sibling_ix);
                    self.set_child(sibling_ix, Some(parent_ix), node_is_left);
                    self.set_child(parent_ix, close_nephew_ix, !node_is_left);
                }
            }
            // Case (terminal): Node has red parent, sibling and both nephews are black.
            else if !parent_black && sibling_black && close_nephew_black && distant_nephew_black {
                self.set_black(parent_ix);
                self.set_red(sibling_ix);
                return;
            }
            // Case: Node has any parent, sibling and distant nephew, but close nephew is red.
            else if sibling_black && distant_nephew_black && !close_nephew_black {
                let close_nephew_ix = close_nephew_ix.unwrap(); // !black_or_nil
                self.set_black(close_nephew_ix);
                self.set_red(sibling_ix);

                let close_newphew_child2 = if node_is_left {
                    self.node(close_nephew_ix).right
                } else {
                    self.node(close_nephew_ix).left
                };
                self.set_child(sibling_ix, close_newphew_child2, node_is_left);
                self.set_child(close_nephew_ix, Some(sibling_ix), !node_is_left);
                self.set_child(parent_ix, Some(close_nephew_ix), !node_is_left);
                self.update_subtree_interval(sibling_ix);
                self.update_subtree_interval(close_nephew_ix);
            }
            // Case (terminal): any parent, black sibling, any close sibling and any red distant nephew.
            else {
                debug_assert!(sibling_black && !distant_nephew_black);
                let distant_nephew_ix = distant_nephew_ix.unwrap(); // !black_or_nil
                                                                    // parent's color -> sibling's color.
                self.colors
                    .set(sibling_ix.get(), self.colors.get(parent_ix.get()));
                self.set_black(parent_ix);
                self.set_black(distant_nephew_ix);
                self.replace_children(parent_ix, sibling_ix);
                self.set_child(parent_ix, close_nephew_ix, !node_is_left);
                self.set_child(sibling_ix, Some(parent_ix), node_is_left);
                return;
            }
        }
    }

    /// Restructure the tree before removing `ix`.
    fn restructure_rm(&mut self, ix: Ix, child_ix: Option<Ix>) {
        if self.is_red(ix) {
            // Both of the children must be NIL.
            debug_assert!(child_ix.is_none());
            // Do nothing.
        } else if !self.is_black_or_nil(child_ix) {
            self.set_red(child_ix.unwrap()); // unwrap: !black_or_nil
                                             // Child will be removed later.
        } else {
            self.restructure_rm_complex_cases(ix);
        }
    }

    pub(super) fn remove_at(&mut self, ix: Option<Ix>) -> Option<V> {
        let ix = match ix {
            Some(ix) => ix,
            None => return None,
        };

        let node = self.node(ix);
        let rm_ix = match node.right {
            None => ix,
            Some(right) => {
                // Searching for a minimal node in the right subtree.
                let mut curr = right;
                loop {
                    match self.node(curr).left {
                        Some(left) => curr = left,
                        None => break curr,
                    }
                }
            }
        };
        if rm_ix != ix {
            self.swap_nodes(ix, rm_ix);
        }

        let rm_node = self.node(rm_ix);
        let child_ix = rm_node.left.or(rm_node.right);
        self.restructure_rm(rm_ix, child_ix);

        match child_ix {
            Some(child_ix) => {
                // Removed node has a child, replace the node with the child and remove the child.
                self.swap_nodes(rm_ix, child_ix);
                self.remove_child(rm_ix, child_ix);
                self.fix_intervals_up(Some(rm_ix));
                Some(self.swap_remove(child_ix))
            }
            None => {
                // Removed node has no child, just remove the node.
                match self.node(rm_ix).parent {
                    Some(parent_ix) => {
                        self.remove_child(parent_ix, rm_ix);
                        self.fix_intervals_up(Some(parent_ix));
                    }
                    None => {
                        debug_assert!(self.len() == 1 && self.root == Some(rm_ix));
                        self.root = None;
                    }
                }
                Some(self.swap_remove(rm_ix))
            }
        }
    }
}
