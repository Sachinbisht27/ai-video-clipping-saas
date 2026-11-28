const menu = document.querySelector('.menu-toggle');
const navigation = document.querySelector('#navigation');
menu.addEventListener('click', () => {
  const open = navigation.classList.toggle('open');
  menu.setAttribute('aria-expanded', String(open));
});
navigation.querySelectorAll('a').forEach(link => link.addEventListener('click', () => {
  navigation.classList.remove('open');
  menu.setAttribute('aria-expanded', 'false');
}));
const choices = document.querySelectorAll('.choice');
choices.forEach(choice => choice.addEventListener('click', () => {
  choices.forEach(item => {
    item.classList.remove('selected');
    item.setAttribute('aria-pressed', 'false');
  });
  choice.classList.add('selected');
  choice.setAttribute('aria-pressed', 'true');
  document.querySelector('#demo-title').textContent = choice.dataset.title;
  document.querySelector('#demo-duration').textContent = choice.dataset.duration;
  document.querySelector('#demo-times').textContent = choice.dataset.times;
  document.querySelector('.demo-note').textContent = `Selected: ${choice.dataset.title}. Interactive example; no video is playing.`;
}));
const waveform = document.querySelector('#waveform');
if (waveform) for (let i = 0; i < 90; i++) {
  const bar = document.createElement('span');
  bar.style.height = `${20 + ((i * 31 + i * i * 7) % 70)}%`;
  waveform.appendChild(bar);
}

const chapters = document.querySelectorAll('.story-chapter');
const stories = [ ['▶','Begin with a conversation','A longer video holds the story.'], ['✳','Find the moments','Suggestions appear with their context.'], ['◒','Shape a complete thought','Review and adjust the boundaries.'], ['↗','Share your moment','The creator chooses what feels right.'] ];
function showStory(index) {
  chapters.forEach((chapter, i) => {
    chapter.classList.toggle('active', i === index);
    chapter.setAttribute('aria-pressed', String(i === index));
  });
  document.querySelector('#story-count').textContent = `0${index + 1} / 04`;
  document.querySelectorAll('.story-scene').forEach((scene, i) => scene.classList.toggle('is-visible', i === index));
  document.querySelector('#story-visual-title').textContent = stories[index][1];
  document.querySelector('#story-visual-detail').textContent = stories[index][2];
}
if (chapters.length) {
  chapters.forEach(chapter => chapter.addEventListener('click', () => showStory(Number(chapter.dataset.story))));
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let scheduled = false;
  function updateStory() {
    scheduled = false;
    if (reducedMotion.matches) return;
    const focusLine = window.innerHeight * 0.52;
    let closest = 0;
    let distance = Infinity;
    chapters.forEach((chapter, index) => {
      const rect = chapter.getBoundingClientRect();
      const nextDistance = Math.abs(rect.top + rect.height / 2 - focusLine);
      if (nextDistance < distance) { distance = nextDistance; closest = index; }
    });
    const section = document.querySelector('.story-layout');
    const bounds = section.getBoundingClientRect();
    if (bounds.top < window.innerHeight && bounds.bottom > 0) {
      showStory(closest);
      const progress = Math.max(0, Math.min(1, (focusLine - bounds.top) / (bounds.height || 1)));
      section.style.setProperty('--story-progress', String(progress));
    }
  }
  function scheduleStory() {
    if (!scheduled) { scheduled = true; requestAnimationFrame(updateStory); }
  }
  window.addEventListener('scroll', scheduleStory, {passive: true});
  window.addEventListener('resize', scheduleStory);
  scheduleStory();
}
// Center the sticky panel using its actual height as text and viewport sizes change.
const storyPanel = document.querySelector('.story-visual');
if (storyPanel && 'ResizeObserver' in window) {
  new ResizeObserver(entries => {
    storyPanel.style.setProperty('--story-panel-height', `${entries[0].borderBoxSize?.[0]?.blockSize ?? storyPanel.offsetHeight}px`);
  }).observe(storyPanel);
}
