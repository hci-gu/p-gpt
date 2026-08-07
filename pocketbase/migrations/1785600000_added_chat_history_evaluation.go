package migrations

import (
	"github.com/pocketbase/pocketbase/core"
	m "github.com/pocketbase/pocketbase/migrations"
)

func init() {
	m.Register(func(app core.App) error {
		collection, err := app.FindCollectionByNameOrId("pbc_1009580068")
		if err != nil {
			return err
		}

		if err := collection.Fields.AddMarshaledJSONAt(4, []byte(`{
			"autogeneratePattern": "",
			"default": "active",
			"hidden": false,
			"id": "textevalstatus1",
			"max": 16,
			"min": 0,
			"name": "status",
			"pattern": "^(active|evaluating|completed)$",
			"presentable": false,
			"primaryKey": false,
			"required": false,
			"system": false,
			"type": "text"
		}`)); err != nil {
			return err
		}

		if err := collection.Fields.AddMarshaledJSONAt(5, []byte(`{
			"hidden": false,
			"id": "dateevalcomp01",
			"max": "",
			"min": "",
			"name": "completed_at",
			"presentable": false,
			"required": false,
			"system": false,
			"type": "date"
		}`)); err != nil {
			return err
		}

		if err := collection.Fields.AddMarshaledJSONAt(6, []byte(`{
			"hidden": false,
			"id": "jsonevalresult1",
			"maxSize": 0,
			"name": "evaluation",
			"presentable": false,
			"required": false,
			"system": false,
			"type": "json"
		}`)); err != nil {
			return err
		}

		return app.Save(collection)
	}, func(app core.App) error {
		collection, err := app.FindCollectionByNameOrId("pbc_1009580068")
		if err != nil {
			return err
		}

		collection.Fields.RemoveById("textevalstatus1")
		collection.Fields.RemoveById("dateevalcomp01")
		collection.Fields.RemoveById("jsonevalresult1")
		return app.Save(collection)
	})
}
