/**
 * Lightweight frames-per-second tracker.
 * Keeps the implementation minimal while satisfying isolated module rules.
 */
export class FPSMeter {
	private lastSample = 0;
	private frameCount = 0;
	private fps = 0;

	tick(now: number = typeof performance !== 'undefined' ? performance.now() : Date.now()): number {
		if (this.lastSample === 0) {
			this.lastSample = now;
			this.frameCount = 0;
			this.fps = 0;
			return this.fps;
		}

		this.frameCount += 1;
		const elapsed = now - this.lastSample;
		if (elapsed >= 1000) {
			this.fps = (this.frameCount * 1000) / elapsed;
			this.frameCount = 0;
			this.lastSample = now;
		}
		return this.fps;
	}

	reset(): void {
		this.lastSample = 0;
		this.frameCount = 0;
		this.fps = 0;
	}

	get value(): number {
		return this.fps;
	}
}

export interface FpsOverlayHandle {
	meter: FPSMeter;
	element: HTMLElement;
	valueElement: HTMLElement;
	frameMsElement: HTMLElement;
}

export const fpsMeter = new FPSMeter();

function ensureOverlayContainer(canvas: HTMLCanvasElement): HTMLElement {
	const doc = canvas.ownerDocument ?? document;
	const fallbackParent = doc.body ?? canvas;
	const parent = (canvas.parentElement as HTMLElement | null) ?? fallbackParent;
	if (parent.contains(canvas)) {
		return parent;
	}
	parent.appendChild(canvas);
	return parent;
}

export function createFpsOverlay(canvas: HTMLCanvasElement): FpsOverlayHandle {
	const container = ensureOverlayContainer(canvas);
	const doc = canvas.ownerDocument ?? document;
	const element = doc.createElement('div');
	element.style.right = '12px';
	element.style.bottom = '12px';
	element.style.padding = '6px 8px 8px 8px';
	element.style.fontFamily = 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
	element.style.fontSize = '12px';
	element.style.lineHeight = '16px';
	element.style.color = '#f5f7fa';
	element.style.background = 'linear-gradient(135deg, rgba(18, 18, 22, 0.72), rgba(36, 40, 48, 0.65))';
	element.style.borderRadius = '12px';
	element.style.border = '1px solid rgba(255, 255, 255, 0.12)';
	element.style.boxShadow = '0 10px 30px rgba(0, 0, 0, 0.35)';
	element.style.pointerEvents = 'none';
	element.style.userSelect = 'none';
	element.style.backdropFilter = 'blur(10px)';
	element.style.setProperty('-webkit-backdrop-filter', 'blur(10px)');
	element.style.display = 'flex';
	element.style.flexDirection = 'column';
	element.style.rowGap = '6px';
	element.style.minWidth = '120px';
	element.style.letterSpacing = '0.25px';
	const title = doc.createElement('div');
	title.textContent = 'FPS Monitor';
	title.style.fontSize = '11px';
	title.style.fontWeight = '600';
	title.style.opacity = '0.8';
	title.style.textTransform = 'uppercase';
	title.style.letterSpacing = '1.2px';
	const valueRow = doc.createElement('div');
	valueRow.style.display = 'flex';
	valueRow.style.alignItems = 'baseline';
	valueRow.style.columnGap = '10px';
	const value = doc.createElement('span');
	value.textContent = '--';
	value.style.fontSize = '24px';
	value.style.fontWeight = '700';
	value.style.minWidth = '44px';
	value.style.textAlign = 'right';
	value.style.color = '#7df9ff';
	const unit = doc.createElement('span');
	unit.textContent = 'fps';
	unit.style.fontSize = '12px';
	unit.style.opacity = '0.8';
	const subRow = doc.createElement('div');
	subRow.style.display = 'flex';
	subRow.style.alignItems = 'center';
	subRow.style.columnGap = '6px';
	subRow.style.opacity = '0.75';
	const msLabel = doc.createElement('span');
	msLabel.textContent = 'Frame time';
	msLabel.style.fontSize = '11px';
	const msValue = doc.createElement('span');
	msValue.textContent = '-- ms';
	msValue.style.fontSize = '11px';
	msValue.style.fontVariantNumeric = 'tabular-nums';
	subRow.appendChild(msLabel);
	subRow.appendChild(msValue);
	valueRow.appendChild(value);
	valueRow.appendChild(unit);
	element.appendChild(title);
	element.appendChild(valueRow);
	element.appendChild(subRow);
	const body = doc.body;
	const isBody = !!body && container === body;
	if (isBody) {
		element.style.position = 'fixed';
	} else {
		element.style.position = 'absolute';
		if (!container.style.position || container.style.position === 'static') {
			container.style.position = 'relative';
		}
	}
	container.appendChild(element);
	return {
		meter: new FPSMeter(),
		element,
		valueElement: value,
		frameMsElement: msValue,
	};
}

export function updateFpsOverlay(handle: FpsOverlayHandle | null | undefined, now?: number): void {
	if (!handle) {
		return;
	}
	const fps = handle.meter.tick(now);
	const frameMs = fps > 0 ? (1000 / fps) : 0;
	handle.valueElement.textContent = fps > 0 ? fps.toFixed(0) : '--';
	handle.frameMsElement.textContent = frameMs > 0 ? `${frameMs.toFixed(1)} ms` : '-- ms';
	let accent = '#4cd964';
	let border = '#4cd96455';
	if (fps <= 30) {
		accent = '#ff6b6b';
		border = '#ff6b6b55';
	} else if (fps <= 55) {
		accent = '#ffd166';
		border = '#ffd16655';
	}
	handle.element.style.borderColor = border;
	handle.valueElement.style.color = accent;
}

export function disposeFpsOverlay(handle: FpsOverlayHandle | null | undefined): void {
	if (!handle) {
		return;
	}
	if (handle.element.parentElement) {
		handle.element.parentElement.removeChild(handle.element);
	}
	handle.meter.reset();
}
