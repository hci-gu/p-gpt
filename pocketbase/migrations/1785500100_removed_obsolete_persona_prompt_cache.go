package migrations

import (
	"github.com/pocketbase/pocketbase/core"
	m "github.com/pocketbase/pocketbase/migrations"
)

func init() {
	m.Register(func(app core.App) error {
		collection, err := app.FindCollectionByNameOrId("pbc_3317324350")
		if err != nil {
			return err
		}

		collection.Fields.RemoveByName("system_prompt")
		collection.Fields.RemoveByName("system_prompt_cache_key")

		return app.Save(collection)
	}, func(app core.App) error {
		return nil
	})
}
