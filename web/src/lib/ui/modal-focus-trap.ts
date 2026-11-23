// Shared keyboard focus-trap Svelte action for modal dialogs (Wave 2).
// Traps Tab / Shift+Tab cycling within the dialog container and restores focus
// to the triggering element on teardown.

const FOCUSABLE_SELECTOR = [
  'a[href]',
  'area[href]',
  'button:not([disabled])',
  'input:not([disabled]):not([type="hidden"])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  'iframe',
  'audio[controls]',
  'video[controls]',
  '[contenteditable]:not([contenteditable="false"])',
  '[tabindex]:not([tabindex="-1"])',
].join(', ');

function getFocusableElements(root: HTMLElement): HTMLElement[] {
  return Array.from(root.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)).filter(
    (el) => !el.hasAttribute('disabled') && el.getAttribute('aria-hidden') !== 'true' && el.offsetParent !== null,
  );
}

export interface ModalFocusTrapOptions {
  /** Whether to move initial focus into the dialog if focus isn't already inside. Default: true. */
  autoFocus?: boolean;
  /** Whether to restore focus to the previously focused element on destroy. Default: true. */
  restoreFocus?: boolean;
}

export function modalFocusTrap(
  node: HTMLElement,
  options: ModalFocusTrapOptions = {},
): { update: (next?: ModalFocusTrapOptions) => void; destroy: () => void } {
  let opts = { autoFocus: true, restoreFocus: true, ...options };
  const previouslyFocused =
    typeof document !== 'undefined' && document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null;

  const ensureInitialFocus = () => {
    if (!opts.autoFocus) return;
    if (typeof document === 'undefined') return;
    if (node.contains(document.activeElement)) return;
    const focusables = getFocusableElements(node);
    if (focusables.length > 0) {
      focusables[0].focus({ preventScroll: true });
    } else if (node.tabIndex >= 0 || node.hasAttribute('tabindex')) {
      node.focus({ preventScroll: true });
    }
  };

  // Defer by a microtask so child onMount focus calls (e.g. confirmBtn.focus()) take precedence.
  queueMicrotask(ensureInitialFocus);

  const handleKeydown = (ev: KeyboardEvent) => {
    if (ev.key !== 'Tab') return;
    const focusables = getFocusableElements(node);
    if (focusables.length === 0) {
      ev.preventDefault();
      return;
    }
    const first = focusables[0];
    const last = focusables[focusables.length - 1];
    const active = document.activeElement as HTMLElement | null;

    if (ev.shiftKey) {
      if (!active || active === first || !node.contains(active)) {
        ev.preventDefault();
        last.focus();
      }
    } else {
      if (!active || active === last || !node.contains(active)) {
        ev.preventDefault();
        first.focus();
      }
    }
  };

  node.addEventListener('keydown', handleKeydown);

  return {
    update(next?: ModalFocusTrapOptions) {
      opts = { autoFocus: true, restoreFocus: true, ...(next ?? {}) };
    },
    destroy() {
      node.removeEventListener('keydown', handleKeydown);
      if (opts.restoreFocus && previouslyFocused && previouslyFocused.isConnected) {
        try {
          previouslyFocused.focus({ preventScroll: true });
        } catch {
          /* ignore focus restoration failure */
        }
      }
    },
  };
}
