communications:
```rust

struct Tick(u64);
struct Clientward {
	when: Tick,
	what: ClientwardWhat
}
enum ClientwardWhat {
	EntityOverwrite {
		id: EntityId,
		entity: Entity,
	},
	Cr
}
```