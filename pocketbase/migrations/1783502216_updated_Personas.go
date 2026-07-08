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

		// update field
		if err := collection.Fields.AddMarshaledJSONAt(4, []byte(`{
			"help": "Profile picture of person",
			"hidden": false,
			"id": "file3311767829",
			"maxSelect": 0,
			"maxSize": 0,
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
		}`)); err != nil {
			return err
		}

		// update field
		if err := collection.Fields.AddMarshaledJSONAt(5, []byte(`{
			"help": "Audio sample used for voice cloning and voice preview",
			"hidden": false,
			"id": "file4077156197",
			"maxSelect": 0,
			"maxSize": 0,
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
		}`)); err != nil {
			return err
		}

		return app.Save(collection)
	}, func(app core.App) error {
		collection, err := app.FindCollectionByNameOrId("pbc_3317324350")
		if err != nil {
			return err
		}

		// update field
		if err := collection.Fields.AddMarshaledJSONAt(4, []byte(`{
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
		}`)); err != nil {
			return err
		}

		// update field
		if err := collection.Fields.AddMarshaledJSONAt(5, []byte(`{
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
		}`)); err != nil {
			return err
		}

		return app.Save(collection)
	})
}
