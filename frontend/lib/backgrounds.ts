export type BackgroundOption = {
  id: string
  label: string
  url: string | null
}

const backgroundModules = import.meta.glob('../assets/background*.png', {
  eager: true,
  import: 'default',
  query: '?url',
}) as Record<string, string>

const numberedBackgrounds = Object.entries(backgroundModules)
  .flatMap(([path, url]) => {
    const match = path.match(/background(\d+)\.png$/)
    if (!match) {
      return []
    }

    const number = Number(match[1])
    return [
      {
        id: `background${number}`,
        label: `Background ${number}`,
        number,
        url,
      },
    ]
  })
  .sort((first, second) => first.number - second.number)
  .map(({ id, label, url }) => ({ id, label, url }))

export const backgroundOptions: BackgroundOption[] = [
  { id: 'none', label: 'No background', url: null },
  ...numberedBackgrounds,
]

export const defaultBackgroundId =
  numberedBackgrounds[0]?.id ?? backgroundOptions[0].id
