/** createElement + class + text in one go. Text is always set via textContent. */
export function el<K extends keyof HTMLElementTagNameMap>(
	tag: K,
	className?: string,
	text?: string,
): HTMLElementTagNameMap[K] {
	const node = document.createElement(tag)
	if (className) node.className = className
	if (text !== undefined) node.textContent = text
	return node
}

export function byId<T extends HTMLElement = HTMLElement>(id: string): T {
	const node = document.getElementById(id)
	if (!node) throw new Error(`Missing element #${id}`)
	return node as T
}
