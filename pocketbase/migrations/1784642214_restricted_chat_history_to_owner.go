package migrations

import (
	"github.com/pocketbase/pocketbase/core"
	m "github.com/pocketbase/pocketbase/migrations"
	"github.com/pocketbase/pocketbase/tools/types"
)

func init() {
	m.Register(func(app core.App) error {
		collection, err := app.FindCollectionByNameOrId("pbc_1009580068")
		if err != nil {
			return err
		}

		collection.Fields.Add(&core.RelationField{
			Name:         "owner",
			CollectionId: "_pb_users_auth_",
			MaxSelect:    1,
			Required:     true,
		})

		const ownerRule = "@request.auth.collectionId = '_pb_users_auth_' && owner = @request.auth.id"
		collection.ListRule = types.Pointer(ownerRule)
		collection.ViewRule = types.Pointer(ownerRule)
		collection.CreateRule = types.Pointer("@request.auth.collectionId = '_pb_users_auth_'")
		collection.UpdateRule = types.Pointer(ownerRule + " && @request.body.owner:changed = false")
		collection.DeleteRule = types.Pointer(ownerRule)

		return app.Save(collection)
	}, func(app core.App) error {
		collection, err := app.FindCollectionByNameOrId("pbc_1009580068")
		if err != nil {
			return err
		}

		collection.Fields.RemoveByName("owner")
		collection.ListRule = types.Pointer("")
		collection.ViewRule = types.Pointer("")
		collection.CreateRule = types.Pointer("")
		collection.UpdateRule = types.Pointer("")
		collection.DeleteRule = types.Pointer("")

		return app.Save(collection)
	})
}
