# neural_ca.py — Fractal Seeded Cellular Automaton with GRU, Emotion, Memory, Mycelial Overlay + Recursive Zoom (v2.0+)

import numpy as np
from keras.layers import GRU, Dense, Input
from keras.models import Model

class NeuralCA:
    def __init__(self, grid_size=32, latent_dim=128, sentient_memory=None, emotional_feedback=None, mycelial_engine=None):
        self.grid_size = grid_size
        self.latent_dim = latent_dim
        self.grid = np.zeros((grid_size, grid_size))
        self.sentient_memory = sentient_memory
        self.emotional_feedback = emotional_feedback
        self.mycelial_engine = mycelial_engine
        self.build_gru_modulator()

    def build_gru_modulator(self):
        inp = Input(shape=(1, self.latent_dim))
        x = GRU(64, return_sequences=False)(inp)
        x = Dense(16, activation='relu')(x)
        out = Dense(9, activation='tanh')(x)  # 3x3 kernel
        self.gru_model = Model(inp, out)

    def seed_from_vector(self, seed_vector):
        norm_seed = np.tanh(seed_vector[:self.grid_size ** 2])
        self.grid = norm_seed.reshape((self.grid_size, self.grid_size))

    def overlay_mycelial_pattern(self):
        if not self.mycelial_engine:
            return
        trail = self.mycelial_engine.echo_query("pattern")
        if trail is not None and len(trail) >= self.grid_size ** 2:
            trail_matrix = np.tanh(trail[:self.grid_size ** 2]).reshape((self.grid_size, self.grid_size))
            self.grid += 0.15 * trail_matrix

    def apply_emotion_tint(self, kernel):
        if self.emotional_feedback and hasattr(self.emotional_feedback, 'current_emotion_vector'):
            emo = self.emotional_feedback.current_emotion_vector()
            if emo is not None and len(emo) >= 9:
                return kernel + 0.1 * emo[:9].reshape((3, 3))
        return kernel

    def step(self, latent_vector):
        # Modulate rule via GRU
        latent_input = latent_vector.reshape((1, 1, self.latent_dim))
        rule_kernel = self.gru_model.predict(latent_input, verbose=0).reshape((3, 3))
        rule_kernel = self.apply_emotion_tint(rule_kernel)

        # Mycelial overlay before update
        self.overlay_mycelial_pattern()

        # Apply CA update rule
        new_grid = np.zeros_like(self.grid)
        padded = np.pad(self.grid, 1, mode='wrap')
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                neighborhood = padded[i:i+3, j:j+3]
                new_value = np.sum(neighborhood * rule_kernel)
                new_grid[i, j] = np.tanh(new_value)

        self.grid = new_grid

        # Feedback to memory
        if self.sentient_memory:
            trace_vector = new_grid.flatten()[:self.latent_dim]
            self.sentient_memory.store_trace("phantom_ca", trace_vector)

    def zoom_pattern(self, level=2):
        """Recursive zoom into center subpattern."""
        center = self.grid_size // 2
        size = self.grid_size // (2 ** level)
        start = max(center - size // 2, 0)
        end = min(start + size, self.grid_size)
        sub = self.grid[start:end, start:end]
        return sub

    def generate(self, steps=10, latent_vector=None):
        outputs = []
        for _ in range(steps):
            if latent_vector is not None:
                self.step(latent_vector)
            outputs.append(np.copy(self.grid))
        return outputs

    # New fractal pattern generation methods
    def generate_fractal_pattern(self, iterations=5, rule_variant="mandelbrot"):
        """Generate complex fractal patterns."""
        if rule_variant == "mandelbrot":
            return self._generate_mandelbrot_pattern(iterations)
        elif rule_variant == "julia":
            return self._generate_julia_pattern(iterations)
        elif rule_variant == "barnsley":
            return self._generate_barnsley_pattern(iterations)
        else:
            return self._generate_mandelbrot_pattern(iterations)

    def _generate_mandelbrot_pattern(self, iterations):
        """Generate Mandelbrot set pattern."""
        grid = np.zeros((self.grid_size, self.grid_size))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Map grid coordinates to complex plane
                x = (i - self.grid_size/2) / (self.grid_size/4)
                y = (j - self.grid_size/2) / (self.grid_size/4)
                c = complex(x, y)
                
                z = 0
                for n in range(iterations):
                    if abs(z) > 2:
                        grid[i, j] = n / iterations
                        break
                    z = z*z + c
                else:
                    grid[i, j] = 1.0
                    
        return grid

    def _generate_julia_pattern(self, iterations):
        """Generate Julia set pattern."""
        grid = np.zeros((self.grid_size, self.grid_size))
        # Julia constant
        c = complex(-0.7, 0.27015)
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Map grid coordinates to complex plane
                x = (i - self.grid_size/2) / (self.grid_size/4)
                y = (j - self.grid_size/2) / (self.grid_size/4)
                z = complex(x, y)
                
                for n in range(iterations):
                    if abs(z) > 2:
                        grid[i, j] = n / iterations
                        break
                    z = z*z + c
                else:
                    grid[i, j] = 1.0
                    
        return grid

    def _generate_barnsley_pattern(self, iterations):
        """Generate Barnsley fern pattern."""
        grid = np.zeros((self.grid_size, self.grid_size))
        
        # Initialize point
        x, y = 0, 0
        
        # Transformation coefficients
        transformations = [
            ([0, 0, 0, 0.16, 0, 0], 0.01),
            ([0.85, 0.04, -0.04, 0.85, 0, 1.6], 0.85),
            ([0.2, -0.26, 0.23, 0.22, 0, 1.6], 0.07),
            ([-0.15, 0.28, 0.26, 0.24, 0, 0.44], 0.07)
        ]
        
        # Generate points
        for _ in range(iterations * 100):
            # Choose transformation randomly based on probabilities
            r = np.random.random()
            cumulative = 0
            for coeffs, prob in transformations:
                cumulative += prob
                if r <= cumulative:
                    a, b, c, d, e, f = coeffs
                    x_new = a * x + b * y + e
                    y_new = c * x + d * y + f
                    x, y = x_new, y_new
                    break
            
            # Map to grid coordinates
            grid_x = int((x + 3) * self.grid_size / 6)
            grid_y = int((y + 1) * self.grid_size / 12)
            
            # Ensure coordinates are within bounds
            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                grid[grid_x, grid_y] = 1.0
                
        return grid

    def fractal_modulate(self, base_pattern, fractal_pattern, blend_factor=0.3):
        """Modulate base pattern with fractal pattern."""
        return (1 - blend_factor) * base_pattern + blend_factor * fractal_pattern

    def generate_complex_stimuli(self, latent_vector, complexity_level=1.0):
        """Generate complex sensory stimuli with fractal enhancement."""
        # Generate base pattern through standard CA
        self.seed_from_vector(latent_vector)
        base_pattern = self.grid.copy()
        
        # Generate fractal pattern based on complexity level
        if complexity_level > 0.7:
            fractal_pattern = self.generate_fractal_pattern(iterations=10, rule_variant="mandelbrot")
        elif complexity_level > 0.4:
            fractal_pattern = self.generate_fractal_pattern(iterations=7, rule_variant="julia")
        else:
            fractal_pattern = self.generate_fractal_pattern(iterations=5, rule_variant="barnsley")
        
        # Modulate patterns
        complex_pattern = self.fractal_modulate(base_pattern, fractal_pattern, complexity_level * 0.5)
        self.grid = complex_pattern
        
        return complex_pattern

# Example usage
if __name__ == "__main__":
    latent = np.random.randn(128)
    ca = NeuralCA(grid_size=32, latent_dim=128)
    ca.seed_from_vector(latent)
    output = ca.generate(steps=5, latent_vector=latent)
    for i, frame in enumerate(output):
        print(f"Step {i}: Grid sum = {np.sum(frame):.3f}")
    
    # Test fractal pattern generation
    mandelbrot = ca.generate_fractal_pattern(iterations=5, rule_variant="mandelbrot")
    julia = ca.generate_fractal_pattern(iterations=5, rule_variant="julia")
    barnsley = ca.generate_fractal_pattern(iterations=5, rule_variant="barnsley")
    print(f"Mandelbrot pattern shape: {mandelbrot.shape}")
    print(f"Julia pattern shape: {julia.shape}")
    print(f"Barnsley pattern shape: {barnsley.shape}")
