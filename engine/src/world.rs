pub use slotmap::DefaultKey;
use slotmap::SlotMap;

use crate::entity::Entity;

pub struct World {
    pub(crate) entities: SlotMap<DefaultKey, Box<dyn Entity + 'static>>,
}

impl Default for World {
    fn default() -> Self {
        Self {
            entities: SlotMap::with_key(),
        }
    }
}

impl World {
    pub fn add_entity<E>(&mut self, entity: E) -> DefaultKey
    where
        E: Entity + 'static,
    {
        self.entities.insert(Box::new(entity))
    }

    pub fn get_entity<E>(&self, id: DefaultKey) -> Option<&E>
    where
        E: Entity + 'static,
    {
        self.entities.get(id).and_then(|e| e.downcast_ref::<E>())
    }

    pub fn get_entity_mut<E>(&mut self, id: DefaultKey) -> Option<&mut E>
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
            .collect::<Vec<_>>()
    }

    pub fn get_entities_mut<E>(&mut self) -> Vec<&mut E>
    where
        E: Entity,
    {
        self.entities
            .values_mut()
            .filter(|value| value.is::<E>())
            .filter_map(|value| value.downcast_mut::<E>())
            .collect::<Vec<_>>()
    }
}
