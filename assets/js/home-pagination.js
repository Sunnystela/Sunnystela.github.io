(() => {
  const pageLinkSelector = '[data-notes-page-link]';
  const postListSelector = '#post-list';
  const paginationSelector = '[data-notes-pagination]';
  let activeRequest;

  if (!window.fetch || !window.DOMParser) return;

  const loadNotesPage = async (url) => {
    const currentList = document.querySelector(postListSelector);
    const currentPagination = document.querySelector(paginationSelector);
    if (!currentList || !currentPagination) return;

    const preservedScrollPosition = window.scrollY;
    activeRequest?.abort();
    activeRequest = new AbortController();
    currentPagination.setAttribute('aria-busy', 'true');

    try {
      const response = await fetch(url, {
        headers: { 'X-Requested-With': 'research-notes-pagination' },
        signal: activeRequest.signal
      });
      if (!response.ok) throw new Error(`Pagination request failed: ${response.status}`);

      const nextDocument = new DOMParser().parseFromString(await response.text(), 'text/html');
      const nextList = nextDocument.querySelector(postListSelector);
      const nextPagination = nextDocument.querySelector(paginationSelector);
      if (!nextList || !nextPagination) throw new Error('Pagination content was not found.');

      currentList.replaceWith(nextList);
      currentPagination.replaceWith(nextPagination);

      window.scrollTo({ top: preservedScrollPosition, left: 0, behavior: 'auto' });
    } catch (error) {
      if (error.name === 'AbortError') return;
      window.location.assign(url);
    }
  };

  document.addEventListener('click', (event) => {
    const link = event.target.closest(pageLinkSelector);
    if (!link || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;

    const destination = new URL(link.dataset.pageUrl, window.location.href);
    if (destination.origin !== window.location.origin) return;

    event.preventDefault();
    event.stopImmediatePropagation();
    loadNotesPage(destination.href);
  }, true);
})();
