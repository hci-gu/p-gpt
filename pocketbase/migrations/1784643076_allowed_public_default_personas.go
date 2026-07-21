package migrations

import (
	"fmt"

	"github.com/pocketbase/pocketbase/core"
	m "github.com/pocketbase/pocketbase/migrations"
	"github.com/pocketbase/pocketbase/tools/types"
)

func init() {
	m.Register(func(app core.App) error {
		collection, err := app.FindCollectionByNameOrId("pbc_3317324350")
		if err != nil {
			return err
		}

		owner, ok := collection.Fields.GetByName("owner").(*core.RelationField)
		if !ok {
			return fmt.Errorf("personas owner field is not a relation")
		}
		owner.Required = false

		const ownerRule = "@request.auth.collectionId = '_pb_users_auth_' && owner = @request.auth.id"
		const readRule = "owner = '' || (" + ownerRule + ")"
		collection.ListRule = types.Pointer(readRule)
		collection.ViewRule = types.Pointer(readRule)
		collection.UpdateRule = types.Pointer(ownerRule + " && @request.body.owner:changed = false")

		return app.Save(collection)
	}, func(app core.App) error {
		collection, err := app.FindCollectionByNameOrId("pbc_3317324350")
		if err != nil {
			return err
		}

		owner, ok := collection.Fields.GetByName("owner").(*core.RelationField)
		if !ok {
			return fmt.Errorf("personas owner field is not a relation")
		}
		owner.Required = true

		const ownerRule = "@request.auth.collectionId = '_pb_users_auth_' && owner = @request.auth.id"
		collection.ListRule = types.Pointer(ownerRule)
		collection.ViewRule = types.Pointer(ownerRule)
		collection.UpdateRule = nil

		return app.Save(collection)
	})
}
