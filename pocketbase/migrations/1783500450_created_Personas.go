package migrations

import (
	"encoding/json"

	"github.com/pocketbase/pocketbase/core"
	m "github.com/pocketbase/pocketbase/migrations"
)

func init() {
	m.Register(func(app core.App) error {
		jsonData := `{
			"createRule": null,
			"deleteRule": null,
			"fields": [
				{
					"autogeneratePattern": "[a-z0-9]{15}",
					"help": "",
					"hidden": false,
					"id": "text3208210256",
					"max": 15,
					"min": 15,
					"name": "id",
					"pattern": "^[a-z0-9]+$",
					"presentable": false,
					"primaryKey": true,
					"required": true,
					"system": true,
					"type": "text"
				},
				{
					"autogeneratePattern": "",
					"help": "Full name",
					"hidden": false,
					"id": "text1579384326",
					"max": 0,
					"min": 3,
					"name": "name",
					"pattern": "",
					"presentable": false,
					"primaryKey": false,
					"required": true,
					"system": false,
					"type": "text"
				},
				{
					"autogeneratePattern": "",
					"help": "Short optional description in selection menu",
					"hidden": false,
					"id": "text1843675174",
					"max": 0,
					"min": 0,
					"name": "description",
					"pattern": "",
					"presentable": false,
					"primaryKey": false,
					"required": false,
					"system": false,
					"type": "text"
				},
				{
					"convertURLs": false,
					"help": "Detailed prompt that used to define the personality. Prompt is prepended to messages for person context",
					"hidden": false,
					"id": "editor3767052345",
					"maxSize": 0,
					"name": "instruction_prompt",
					"presentable": false,
					"required": true,
					"system": false,
					"type": "editor"
				},
				{
					"help": "Profile picture of person",
					"hidden": false,
					"id": "file3311767829",
					"maxSelect": 0,
					"maxSize": 10,
					"mimeTypes": [
						"image/png",
						"image/jpeg",
						"image/webp"
					],
					"name": "profile_picture",
					"presentable": false,
					"protected": false,
					"required": false,
					"system": false,
					"thumbs": null,
					"type": "file"
				},
				{
					"help": "Audio sample used for voice cloning and voice preview",
					"hidden": false,
					"id": "file4077156197",
					"maxSelect": 0,
					"maxSize": 10,
					"mimeTypes": [
						"audio/mpeg",
						"audio/wav"
					],
					"name": "audio_sample",
					"presentable": false,
					"protected": false,
					"required": false,
					"system": false,
					"thumbs": null,
					"type": "file"
				},
				{
					"hidden": false,
					"id": "autodate2990389176",
					"name": "created",
					"onCreate": true,
					"onUpdate": false,
					"presentable": false,
					"system": false,
					"type": "autodate"
				},
				{
					"hidden": false,
					"id": "autodate3332085495",
					"name": "updated",
					"onCreate": true,
					"onUpdate": true,
					"presentable": false,
					"system": false,
					"type": "autodate"
				}
			],
			"id": "pbc_3317324350",
			"indexes": [],
			"listRule": null,
			"name": "Personas",
			"system": false,
			"type": "base",
			"updateRule": null,
			"viewRule": null
		}`

		collection := &core.Collection{}
		if err := json.Unmarshal([]byte(jsonData), &collection); err != nil {
			return err
		}

		return app.Save(collection)
	}, func(app core.App) error {
		collection, err := app.FindCollectionByNameOrId("pbc_3317324350")
		if err != nil {
			return err
		}

		return app.Delete(collection)
	})
}
