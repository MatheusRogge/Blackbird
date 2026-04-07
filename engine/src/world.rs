use slotmap::SlotMap;

use crate::entity::{Entity, EntityKey};

pub struct World {
    pub(crate) entities: SlotMap<EntityKey, Box<dyn Entity + 'static>>,
}

pub struct EntityHandle<'a, E: Entity + 'static> {
    pub id: EntityKey,
    pub entity: &'a E,
}

impl Default for World {
    fn default() -> Self {
        Self {
            entities: SlotMap::with_key(),
        }
    }
}

impl World {
    pub fn add_entity<E>(&mut self, entity: E) -> EntityKey
    where
        E: Entity + 'static,
    {
        self.entities.insert(Box::new(entity))
    }

    pub fn get_entity<E>(&self, id: EntityKey) -> Option<&E>
    where
        E: Entity + 'static,
    {
        self.entities.get(id).and_then(|e| e.downcast_ref::<E>())
    }

    pub fn get_entity_mut<E>(&mut self, id: EntityKey) -> Option<&mut E>
    where
        E: Entity + 'static,
    {
        self.entities
            .get_mut(id)
            .and_then(|e| e.downcast_mut::<E>())
    }

    pub fn get_entities<E>(&self) -> Vec<&E>
    where
        E: Entity,
    {
        self.entities
            .values()
            .filter(|value| value.is::<E>())
            .filter_map(|value| value.downcast_ref::<E>())
            .collect()
    }

    pub fn get_entity_handles<'a, E>(&'a self) -> Vec<EntityHandle<'a, E>>
    where
        E: Entity,
    {
        self.entities
            .iter()
            .filter(|(_key, value)| value.is::<E>())
            .filter_map(|(key, value)| {
                value
                    .downcast_ref::<E>()
                    .map(|e| EntityHandle { id: key, entity: e })
            })
            .collect()
    }

    pub fn get_entities_mut<E>(&mut self) -> Vec<&mut E>
    where
        E: Entity,
    {
        self.entities
            .values_mut()
            .filter(|value| value.is::<E>())
            .filter_map(|value| value.downcast_mut::<E>())
            .collect()
    }
}
