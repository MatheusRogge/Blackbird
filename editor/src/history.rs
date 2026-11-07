//! Undo / redo over snapshots of the editable scene state. The UI commits the current snapshot
//! whenever no interaction is in progress, so a whole gizmo drag or typed name is one step.

/// Oldest steps are dropped beyond this.
const MAX_STEPS: usize = 200;

pub struct History<T> {
    undo: Vec<T>,
    redo: Vec<T>,
    /// The state as of the last commit; `None` until the first one after a reset.
    committed: Option<T>,
}

impl<T> Default for History<T> {
    fn default() -> Self {
        Self { undo: Vec::new(), redo: Vec::new(), committed: None }
    }
}

impl<T: Clone + PartialEq> History<T> {
    /// Forgets everything; the next commit becomes the starting point (e.g. after a scene load).
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Records `now` as a step if it differs from the last commit.
    pub fn commit(&mut self, now: T) {
        match &mut self.committed {
            None => self.committed = Some(now),
            Some(previous) if *previous != now => {
                self.undo.push(std::mem::replace(previous, now));
                if self.undo.len() > MAX_STEPS {
                    self.undo.remove(0);
                }
                self.redo.clear();
            }
            Some(_) => {}
        }
    }

    /// Commits `now`, then steps back. Returns the state to restore.
    pub fn undo(&mut self, now: T) -> Option<T> {
        self.commit(now);
        let previous = self.undo.pop()?;
        if let Some(current) = self.committed.replace(previous.clone()) {
            self.redo.push(current);
        }
        Some(previous)
    }

    /// Commits `now`, then steps forward again. Returns the state to restore.
    pub fn redo(&mut self, now: T) -> Option<T> {
        self.commit(now);
        let next = self.redo.pop()?;
        if let Some(current) = self.committed.replace(next.clone()) {
            self.undo.push(current);
        }
        Some(next)
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn undo_and_redo_walk_committed_steps() {
        let mut history = History::default();
        history.commit(0);
        history.commit(1);
        history.commit(1);
        history.commit(2);
        assert_eq!(history.undo(2), Some(1));
        assert_eq!(history.undo(1), Some(0));
        assert_eq!(history.undo(0), None);
        assert_eq!(history.redo(0), Some(1));
        assert_eq!(history.redo(1), Some(2));
        assert_eq!(history.redo(2), None);
    }

    #[test]
    fn uncommitted_change_is_committed_before_undo() {
        let mut history = History::default();
        history.commit(0);
        // Changed but not yet committed (e.g. Ctrl+Z right after a drag): undo returns to 0.
        assert_eq!(history.undo(5), Some(0));
        assert_eq!(history.redo(0), Some(5));
    }

    #[test]
    fn a_new_change_clears_redo() {
        let mut history = History::default();
        history.commit(0);
        history.commit(1);
        history.undo(1);
        history.commit(7);
        assert!(!history.can_redo());
        assert_eq!(history.undo(7), Some(0));
    }

    #[test]
    fn reset_starts_over_from_the_next_commit() {
        let mut history = History::default();
        history.commit(0);
        history.commit(1);
        history.reset();
        history.commit(9);
        assert!(!history.can_undo());
    }
}
