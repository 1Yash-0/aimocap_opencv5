(() => {
  const minimumReadyState = 3;
  const startTimeoutMs = 15000;

  function waitUntilPlayable(videos) {
    if (videos.every((video) => video.readyState >= minimumReadyState)) return Promise.resolve();

    videos.forEach((video) => {
      video.preload = 'auto';
      if (video.readyState === 0) video.load();
    });

    return new Promise((resolve, reject) => {
      let settled = false;
      const cleanup = () => {
        clearTimeout(timer);
        videos.forEach((video) => {
          video.removeEventListener('canplay', checkReady);
          video.removeEventListener('error', fail);
        });
      };
      const finish = (error) => {
        if (settled) return;
        settled = true;
        cleanup();
        if (error) reject(error);
        else resolve();
      };
      const checkReady = () => {
        if (videos.every((video) => video.readyState >= minimumReadyState)) finish();
      };
      const fail = () => finish(new Error('A video could not be loaded.'));
      const timer = setTimeout(() => finish(new Error('Video loading timed out.')), startTimeoutMs);

      videos.forEach((video) => {
        video.addEventListener('canplay', checkReady);
        video.addEventListener('error', fail);
      });
      checkReady();
    });
  }

  document.querySelectorAll('[data-video-group]').forEach((group) => {
    const videos = Array.from(group.querySelectorAll('video'));
    const controls = group.querySelector('[data-video-controls]');
    const button = group.querySelector('[data-play-all]');
    const status = group.querySelector('[data-video-status]');
    if (videos.length < 2 || !controls || !button || !status) return;

    let starting = false;
    let started = false;
    let attempt = 0;
    controls.hidden = false;
    button.hidden = false;
    status.textContent = 'Use Play all to start the videos together.';

    const isPlaying = (video) => !video.paused && !video.ended;
    const updateButton = () => {
      if (starting) {
        button.textContent = 'Loading…';
        button.disabled = true;
        return;
      }
      button.disabled = false;
      if (videos.some(isPlaying)) button.textContent = 'Pause all';
      else if (videos.every((video) => video.ended)) button.textContent = 'Replay all';
      else if (started) button.textContent = 'Resume all';
      else button.textContent = 'Play all';
    };
    videos.forEach((video) => {
      video.muted = true;
      video.playsInline = true;
      ['play', 'pause', 'ended'].forEach((eventName) => video.addEventListener(eventName, updateButton));
      video.addEventListener('error', () => {
        status.textContent = 'A video could not be played. Check the video and try again.';
        videos.forEach((item) => item.pause());
        updateButton();
      });
    });

    button.addEventListener('click', async () => {
      if (starting) return;
      if (videos.some(isPlaying)) {
        videos.forEach((video) => video.pause());
        status.textContent = 'All videos paused.';
        updateButton();
        return;
      }

      const reset = !started || videos.every((video) => video.ended);
      const activeVideos = reset ? videos : videos.filter((video) => !video.ended);
      if (reset) videos.forEach((video) => { video.currentTime = 0; });
      videos.forEach((video) => {
        video.muted = true;
        video.playsInline = true;
      });

      starting = true;
      const thisAttempt = ++attempt;
      let timedOut = false;
      let timer;
      updateButton();
      status.textContent = 'Preparing videos…';
      try {
        await waitUntilPlayable(activeVideos);
        const playPromises = activeVideos.map((video) => Promise.resolve(video.play()).then(() => {
          if (timedOut && attempt === thisAttempt) video.pause();
        }));
        const results = await Promise.race([
          Promise.allSettled(playPromises),
          new Promise((_, reject) => {
            timer = setTimeout(() => {
              timedOut = true;
              reject(new Error('Playback start timed out.'));
            }, startTimeoutMs);
          }),
        ]);
        clearTimeout(timer);
        const failed = results.find((result) => result.status === 'rejected');
        if (failed) throw failed.reason;
        started = true;
        status.textContent = 'Videos playing together.';
      } catch {
        timedOut = true;
        videos.forEach((video) => video.pause());
        status.textContent = 'Could not start all videos. Check the videos and try again.';
      } finally {
        clearTimeout(timer);
        starting = false;
        updateButton();
      }
    });

    updateButton();
  });
})();
