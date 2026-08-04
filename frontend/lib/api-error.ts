const formatDetail = (detail: unknown): string | null => {
  if (typeof detail === 'string' && detail.trim()) {
    return detail.trim()
  }

  if (Array.isArray(detail)) {
    const messages = detail.flatMap((item) => {
      if (typeof item !== 'object' || item === null) {
        return []
      }

      const message = 'msg' in item && typeof item.msg === 'string' ? item.msg : null
      const location =
        'loc' in item && Array.isArray(item.loc)
          ? item.loc
              .filter(
                (part: unknown): part is string | number =>
                  typeof part === 'string' || typeof part === 'number'
              )
              .join('.')
          : ''

      return message ? [`${location ? `${location}: ` : ''}${message}`] : []
    })

    return messages.length ? messages.join('; ') : null
  }

  return null
}

export const getApiErrorMessage = async (
  response: Response,
  stage: string
) => {
  let detail: string | null = null

  try {
    const contentType = response.headers.get('content-type')
    if (contentType?.includes('application/json')) {
      const payload: unknown = await response.json()
      if (typeof payload === 'object' && payload !== null && 'detail' in payload) {
        detail = formatDetail(payload.detail)
      }
    } else {
      const text = await response.text()
      detail = text.trim() || null
    }
  } catch {
    // Fall back to the HTTP status when the error body is malformed.
  }

  return `${stage}: ${detail ?? `request failed with status ${response.status}`}`
}
