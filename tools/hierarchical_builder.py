"""
Hierarchical Builder Module

Provides a TensorFlow/PyTorch-style API for building hierarchical learning systems.
Makes KATO configuration transparent and educational.

Key Classes:
- HierarchicalLayer: Configuration for one layer
- HierarchicalBuilder: Builder pattern for creating hierarchy
- HierarchicalModel: Built model with layers and tokenizer

Example Usage:
```python
from tools.hierarchical_builder import HierarchicalBuilder

# Build hierarchy with explicit layer configuration
hierarchy = HierarchicalBuilder(tokenizer_name='gpt2')

hierarchy.add_layer(
    name='node0',
    chunk_size=15,
    max_predictions=10,
    prediction_field='name',
    recall_threshold=0.6,
    capture_metadata=False  # Don't capture metadata at this layer
)

hierarchy.add_layer(
    name='node1',
    chunk_size=15,
    max_predictions=8,
    prediction_field='name',
    capture_metadata=True  # Capture metadata at this layer
)

model = hierarchy.build()
model.summary()
```
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from transformers import AutoTokenizer
from tools.kato_client import KATOClient


@dataclass
class HierarchicalLayer:
    """
    Configuration for one hierarchical layer.

    Attributes:
        name: Layer identifier (e.g., 'node0', 'node1')
        chunk_size: Number of inputs per chunk
        max_predictions: Maximum predictions to request from KATO
        prediction_field: Which field to extract from predictions ('name', 'pattern_name', etc.)
        recall_threshold: Pattern matching strictness (0.0-1.0)
        stm_mode: Short-term memory mode ('CLEAR' or 'ROLLING')
        max_pattern_length: Auto-learning threshold (0 = manual only)
        process_predictions: Whether KATO computes predictions
        rolling_window_size: Size of rolling window (if stm_mode='ROLLING')
        max_stm_size: Maximum STM size
        capture_metadata: Whether this layer should capture source metadata
        base_url: KATO server URL

    Runtime Attributes (set during build):
        client: KATOClient instance
        level: Layer index in hierarchy
    """

    name: str
    chunk_size: int
    max_predictions: int
    prediction_field: str = 'name'
    recall_threshold: float = 0.6
    stm_mode: str = 'CLEAR'
    max_pattern_length: int = 0
    process_predictions: bool = False
    rolling_window_size: Optional[int] = None
    max_stm_size: Optional[int] = None
    capture_metadata: bool = False
    base_url: str = 'http://kato:8000'

    # Runtime attributes (set during build)
    client: Optional[KATOClient] = None
    level: int = 0

    def should_capture_metadata(self) -> bool:
        """Check if this layer should capture metadata."""
        return self.capture_metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'level': self.level,
            'chunk_size': self.chunk_size,
            'max_predictions': self.max_predictions,
            'prediction_field': self.prediction_field,
            'recall_threshold': self.recall_threshold,
            'stm_mode': self.stm_mode,
            'max_pattern_length': self.max_pattern_length,
            'process_predictions': self.process_predictions,
            'rolling_window_size': self.rolling_window_size,
            'max_stm_size': self.max_stm_size,
            'capture_metadata': self.capture_metadata,
            'base_url': self.base_url
        }


class HierarchicalBuilder:
    """
    Builder for creating hierarchical learning systems.

    Provides a familiar API similar to TensorFlow/PyTorch for adding layers.

    Example:
    ```python
    builder = HierarchicalBuilder(tokenizer_name='gpt2')
    builder.add_layer('node0', chunk_size=15, max_predictions=10)
    builder.add_layer('node1', chunk_size=15, max_predictions=8)
    model = builder.build()
    ```
    """

    def __init__(self, tokenizer_name: str = 'gpt2', base_url: str = 'http://kato:8000'):
        """
        Initialize builder.

        Args:
            tokenizer_name: HuggingFace tokenizer name
            base_url: KATO server URL
        """
        self.tokenizer_name = tokenizer_name
        self.base_url = base_url
        self.layers: List[HierarchicalLayer] = []

    def add_layer(
        self,
        name: str,
        chunk_size: int,
        max_predictions: int,
        prediction_field: str = 'name',
        recall_threshold: float = 0.6,
        stm_mode: str = 'CLEAR',
        max_pattern_length: int = 0,
        process_predictions: bool = False,
        rolling_window_size: Optional[int] = None,
        max_stm_size: Optional[int] = None,
        capture_metadata: bool = False,
        base_url: Optional[str] = None
    ) -> 'HierarchicalBuilder':
        """
        Add a layer to the hierarchy.

        Args:
            name: Layer identifier (e.g., 'node0')
            chunk_size: Number of inputs per chunk
            max_predictions: Maximum predictions from KATO
            prediction_field: Which prediction field to extract ('name', 'pattern_name')
            recall_threshold: Pattern matching strictness (0.0-1.0)
            stm_mode: 'CLEAR' or 'ROLLING'
            max_pattern_length: Auto-learning threshold (0 = manual only)
            process_predictions: Whether KATO computes predictions
            rolling_window_size: Size of rolling window (if stm_mode='ROLLING')
            max_stm_size: Maximum STM size
            capture_metadata: Whether this layer should capture source metadata
            base_url: KATO server URL (defaults to builder's base_url)

        Returns:
            self for method chaining
        """
        layer = HierarchicalLayer(
            name=name,
            chunk_size=chunk_size,
            max_predictions=max_predictions,
            prediction_field=prediction_field,
            recall_threshold=recall_threshold,
            stm_mode=stm_mode,
            max_pattern_length=max_pattern_length,
            process_predictions=process_predictions,
            rolling_window_size=rolling_window_size,
            max_stm_size=max_stm_size,
            capture_metadata=capture_metadata,
            base_url=base_url or self.base_url
        )

        self.layers.append(layer)
        return self  # For chaining

    def build(self, verbose: bool = True) -> 'HierarchicalModel':
        """
        Build the hierarchy and initialize KATO clients.

        Args:
            verbose: Print build information

        Returns:
            HierarchicalModel instance ready for training
        """
        if len(self.layers) == 0:
            raise ValueError("No layers added. Use add_layer() to add at least one layer.")

        if verbose:
            print(f"{'='*80}")
            print(f"BUILDING HIERARCHICAL MODEL")
            print(f"{'='*80}")
            print(f"Tokenizer: {self.tokenizer_name}")
            print(f"Layers: {len(self.layers)}")
            print(f"Base URL: {self.base_url}")
            print()

        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if verbose:
            print(f"✓ Tokenizer loaded: {self.tokenizer_name}")

        # Initialize KATO clients for each layer
        for i, layer in enumerate(self.layers):
            layer.level = i
            layer.client = KATOClient(
                base_url=layer.base_url,
                node_id=layer.name,
                max_pattern_length=layer.max_pattern_length,
                recall_threshold=layer.recall_threshold,
                max_predictions=layer.max_predictions,
                stm_mode=layer.stm_mode,
                process_predictions=layer.process_predictions
            )

            if verbose:
                print(f"✓ Layer {i} ({layer.name}): KATO client initialized")

        if verbose:
            print(f"\n{'='*80}")
            print(f"MODEL BUILD COMPLETE")
            print(f"{'='*80}\n")

        return HierarchicalModel(self.layers, tokenizer, self.tokenizer_name)


class HierarchicalModel:
    """
    Built hierarchical model with layers and tokenizer.

    Provides methods for processing data through the hierarchy.
    """

    def __init__(self, layers: List[HierarchicalLayer], tokenizer, tokenizer_name: str):
        """
        Initialize model.

        Args:
            layers: List of configured HierarchicalLayer instances
            tokenizer: HuggingFace tokenizer
            tokenizer_name: Name of tokenizer
        """
        self.layers = layers
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        self.num_layers = len(layers)

    def summary(self):
        """Print model architecture summary."""
        print(f"\n{'='*80}")
        print(f"HIERARCHICAL MODEL SUMMARY")
        print(f"{'='*80}")
        print(f"Tokenizer: {self.tokenizer_name}")
        print(f"Total Layers: {self.num_layers}")
        print(f"\n{'Layer':<10} {'Name':<10} {'Chunk':<8} {'MaxPred':<8} {'Recall':<8} {'STM Mode':<10} {'Metadata':<10}")
        print(f"{'-'*80}")

        for i, layer in enumerate(self.layers):
            metadata_str = 'Yes' if layer.should_capture_metadata() else 'No'
            print(
                f"{i:<10} "
                f"{layer.name:<10} "
                f"{layer.chunk_size:<8} "
                f"{layer.max_predictions:<8} "
                f"{layer.recall_threshold:<8.2f} "
                f"{layer.stm_mode:<10} "
                f"{metadata_str:<10}"
            )

        print(f"{'='*80}\n")

        # Show receptive field calculation
        print("Receptive Fields (token coverage):")
        receptive_field = 1
        for i, layer in enumerate(self.layers):
            receptive_field *= layer.chunk_size
            print(f"  {layer.name}: {receptive_field:,} tokens")
        print()

    def get_layer(self, index: int) -> HierarchicalLayer:
        """Get layer by index."""
        if index < 0 or index >= self.num_layers:
            raise ValueError(f"Layer index {index} out of range [0, {self.num_layers-1}]")
        return self.layers[index]

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text.

        Args:
            text: Input text

        Returns:
            List of token strings
        """
        return self.tokenizer.tokenize(text)

    def chunk_tokens(self, tokens: List[str], chunk_size: int) -> List[List[str]]:
        """
        Split tokens into fixed-length chunks.

        Args:
            tokens: List of token strings
            chunk_size: Number of tokens per chunk

        Returns:
            List of token chunks
        """
        chunks = []
        for i in range(0, len(tokens), chunk_size):
            chunk = tokens[i:i + chunk_size]
            chunks.append(chunk)
        return chunks

    def decode_tokens(self, tokens: List[str]) -> str:
        """
        Decode tokens back to text.

        Args:
            tokens: List of token strings

        Returns:
            Decoded text
        """
        return self.tokenizer.convert_tokens_to_string(tokens)

    def clear_all_stm(self):
        """Clear short-term memory for all layers."""
        for layer in self.layers:
            layer.client.clear_stm()

    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration.

        Returns:
            Dictionary with model configuration
        """
        return {
            'tokenizer_name': self.tokenizer_name,
            'num_layers': self.num_layers,
            'layers': [layer.to_dict() for layer in self.layers]
        }

    def __repr__(self) -> str:
        return f"HierarchicalModel(layers={self.num_layers}, tokenizer={self.tokenizer_name})"


# Utility functions for explicit KATO operations (used in notebooks)

def process_chunk_at_layer(
    chunk: List[str],
    kato_client: KATOClient,
    metadata: Optional[Dict[str, Any]] = None,
    verbose: bool = False
) -> str:
    """
    Process one chunk at a layer using explicit KATO calls.

    Educational function that shows exact KATO API calls.

    Args:
        chunk: List of symbols (tokens or pattern names)
        kato_client: KATO client for this layer
        metadata: Optional metadata to attach to observations
        verbose: Print progress

    Returns:
        Learned pattern name
    """
    # Clear STM
    kato_client.clear_stm()

    if verbose:
        print(f"  Cleared STM")

    # Build observations (one symbol per event)
    observations = []
    for symbol in chunk:
        obs = {'strings': [symbol]}
        if metadata:
            obs['metadata'] = metadata
        observations.append(obs)

    if verbose:
        print(f"  Built {len(observations)} observations")

    # Send to KATO with learn_at_end=True
    result = kato_client.observe_sequence(
        observations=observations,
        learn_at_end=True
    )

    # Extract pattern name
    pattern_name = result.get('final_learned_pattern') or result.get('pattern_name', 'UNKNOWN')

    if verbose:
        print(f"  Learned pattern: {pattern_name[:50]}...")

    return pattern_name


def accumulate_in_stm(
    symbols: List[str],
    kato_client: KATOClient,
    metadata: Optional[Dict[str, Any]] = None,
    verbose: bool = False
) -> int:
    """
    Accumulate symbols in STM without learning.

    Educational function that shows explicit KATO calls.

    Args:
        symbols: List of symbols to accumulate
        kato_client: KATO client for this layer
        metadata: Optional metadata to attach
        verbose: Print progress

    Returns:
        Number of symbols accumulated
    """
    for symbol in symbols:
        if metadata:
            kato_client.observe(strings=[symbol], metadata=metadata)
        else:
            kato_client.observe(strings=[symbol])

    if verbose:
        print(f"  Accumulated {len(symbols)} symbols in STM")

    return len(symbols)


def learn_from_stm(
    kato_client: KATOClient,
    verbose: bool = False
) -> str:
    """
    Trigger learning from current STM contents.

    Educational function that shows explicit KATO calls.

    Args:
        kato_client: KATO client for this layer
        verbose: Print progress

    Returns:
        Learned pattern name
    """
    result = kato_client.learn()
    pattern_name = result.get('pattern_name') or result.get('final_learned_pattern', 'UNKNOWN')

    if verbose:
        print(f"  Learned pattern: {pattern_name[:50]}...")

    return pattern_name


def extract_prediction_field(
    predictions: List[Dict[str, Any]],
    field_name: str = 'name',
    max_predictions: Optional[int] = None
) -> List[str]:
    """
    Extract specific field from prediction objects.

    Args:
        predictions: List of prediction dictionaries
        field_name: Field to extract ('name', 'pattern_name', etc.)
        max_predictions: Limit number of predictions

    Returns:
        List of extracted values
    """
    if max_predictions:
        predictions = predictions[:max_predictions]

    extracted = []
    for pred in predictions:
        value = pred.get(field_name)
        if value:
            extracted.append(value)

    return extracted
