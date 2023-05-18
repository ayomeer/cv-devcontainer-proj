class BlitManager:
    def __init__(self, canvas, animated_artists=()):
        self.canvas = canvas
        self._artists = []
        self._bg = None
 
        for a in animated_artists:
            self.add_artist(a)
 
        self.cid = canvas.mpl_connect("draw_event", self.on_draw)
 
 
    def on_draw(self, event):
        """Callback to register with 'draw_event'."""
        if event is not None:
            if event.canvas != self.canvas:
                raise RuntimeError
        self._bg = self.canvas.copy_from_bbox(self.canvas.figure.bbox)
        self._draw_animated()
 
 
    def _draw_animated(self):
        """Draw all of the animated artists."""
        fig = self.canvas.figure
        for a in self._artists:
            fig.draw_artist(a)
 
 
    def add_artist(self, art):
        """Add a new Artist object to the Blit Manager"""
        if art.figure != self.canvas.figure:
            raise RuntimeError
        art.set_animated(True)
        self._artists.append(art)
 
 
    def update(self):
        """Update the screen with animated artists."""
     
        if self._bg is None:
            self.on_draw(None)
        else:
            # restore the background
            self.canvas.restore_region(self._bg)
            # draw all of the animated artists
            self._draw_animated()
            # update the GUI state
            self.canvas.blit(self.canvas.figure.bbox)
        # let the GUI event loop process anything it has to do
        self.canvas.flush_events()