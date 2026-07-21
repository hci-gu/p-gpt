import PocketBase from 'pocketbase'

const pocketBaseEndpoint = (
  import.meta.env.VITE_POCKETBASE_ENDPOINT ?? 'http://127.0.0.1:8090'
).replace(/\/$/, '')

export const pb = new PocketBase(pocketBaseEndpoint)
