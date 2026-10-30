/// Span of pass indices (inclusive) during which a transient texture is live.
#[derive(Clone, Debug)]
pub struct Lifetime<K> {
    pub key: K,
    pub first: usize,
    pub last: usize,
}

/// Maps each lifetime to a physical slot. Two lifetimes share a slot only if
/// their keys match and one ends strictly before the other starts, so a pass
/// never reads and writes the same physical texture through different ids.
/// Returns the slot per lifetime (input order) and the number of slots.
pub fn assign<K: PartialEq>(lifetimes: &[Lifetime<K>]) -> (Vec<usize>, usize) {
    let mut by_start: Vec<usize> = (0..lifetimes.len()).collect();
    by_start.sort_by_key(|&i| lifetimes[i].first);

    let mut slot_last_use: Vec<(usize, usize)> = Vec::new();
    let mut slots = vec![0; lifetimes.len()];

    for i in by_start {
        let lifetime = &lifetimes[i];
        let free = slot_last_use.iter().position(|&(owner, last)| {
            last < lifetime.first && lifetimes[owner].key == lifetime.key
        });

        slots[i] = match free {
            Some(slot) => {
                slot_last_use[slot] = (i, lifetime.last);
                slot
            }
            None => {
                slot_last_use.push((i, lifetime.last));
                slot_last_use.len() - 1
            }
        };
    }

    (slots, slot_last_use.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lt(key: u32, first: usize, last: usize) -> Lifetime<u32> {
        Lifetime { key, first, last }
    }

    #[test]
    fn disjoint_same_key_share() {
        let (slots, count) = assign(&[lt(0, 0, 1), lt(0, 2, 3)]);
        assert_eq!(slots, [0, 0]);
        assert_eq!(count, 1);
    }

    #[test]
    fn touching_lifetimes_do_not_share() {
        let (slots, count) = assign(&[lt(0, 0, 2), lt(0, 2, 3)]);
        assert_eq!(slots, [0, 1]);
        assert_eq!(count, 2);
    }

    #[test]
    fn different_keys_never_share() {
        let (_, count) = assign(&[lt(0, 0, 0), lt(1, 1, 1)]);
        assert_eq!(count, 2);
    }

    #[test]
    fn reuses_slots_regardless_of_input_order() {
        let (slots, count) = assign(&[lt(0, 4, 5), lt(0, 0, 1), lt(0, 2, 3)]);
        assert_eq!(slots, [0, 0, 0]);
        assert_eq!(count, 1);
    }
}
