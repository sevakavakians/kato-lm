#!/usr/bin/env python3
"""
Test script for hierarchical concept learning demonstration.
"""

import sys
sys.path.insert(0, '/Users/sevakavakians/PROGRAMMING/kato-notebooks')

from kato_hierarchical_streaming import (
    CorpusSegmenter,
    HierarchicalConceptLearner,
)

def main():
    """Run the hierarchical concept learning demonstration."""

    # Sample text for demonstration
    sample_text = """
Chapter 1: Introduction to Artificial Intelligence

Artificial intelligence (AI) is transforming the world. Machine learning algorithms can now recognize patterns in vast amounts of data. Deep learning networks have achieved remarkable results in image recognition and natural language processing.

The field of AI has grown rapidly over the past decade. Researchers continue to push the boundaries of what's possible. New architectures and training methods emerge regularly.

Chapter 2: Machine Learning Fundamentals

Machine learning is a subset of artificial intelligence. It focuses on algorithms that can learn from data without explicit programming. Supervised learning uses labeled examples to train models.

Unsupervised learning finds patterns in unlabeled data. Reinforcement learning trains agents through trial and error. Each approach has its own strengths and applications.
"""

    print("\n" + "="*80)
    print("HIERARCHICAL CONCEPT LEARNING DEMONSTRATION")
    print("="*80)

    # Step 1: Segment the text
    print("\n[Step 1] Segmenting text into hierarchical structure...")
    segmenter = CorpusSegmenter()
    book = segmenter.segment_book(
        sample_text,
        book_metadata={'title': 'AI Primer', 'author': 'Demo'}
    )

    print(f"✓ Segmented into:")
    print(f"  - {len(book['chapters'])} chapters")
    total_paragraphs = sum(len(ch['paragraphs']) for ch in book['chapters'])
    total_sentences = sum(
        len(p['sentences'])
        for ch in book['chapters']
        for p in ch['paragraphs']
    )
    print(f"  - {total_paragraphs} paragraphs")
    print(f"  - {total_sentences} sentences")

    # Step 2: Initialize hierarchical learner
    print("\n[Step 2] Initializing hierarchical concept learner...")
    print("  Using KATO server at: http://localhost:8000")

    learner = HierarchicalConceptLearner(
        base_url="http://localhost:8000",
        tokenizer_name="gpt2"
    )

    # Step 3: Learn the book hierarchically
    print("\n[Step 3] Learning book hierarchically...")
    corpus = {'books': [book]}
    learner.process_corpus(corpus, verbose=True)

    # Step 4: Display detailed statistics
    print("\n[Step 4] Detailed Statistics:")
    print("="*80)
    stats = learner.get_stats()
    print(f"\nLearning Progress:")
    print(f"  Sentences learned: {stats['sentences_learned']:,}")
    print(f"  Paragraphs learned: {stats['paragraphs_learned']:,}")
    print(f"  Chapters learned: {stats['chapters_learned']:,}")
    print(f"  Books learned: {stats['books_learned']:,}")
    print(f"  Elapsed time: {stats['elapsed_formatted']}")

    # Note: Node stats from KATO client may have different format
    print(f"\nPattern Counts by Level:")
    for level in ['node0', 'node1', 'node2', 'node3']:
        patterns = stats['patterns_by_level'][level]
        print(f"  {level.upper()}: {len(patterns)} patterns")

    # Step 5: Show pattern hierarchy
    print("\n[Step 5] Pattern Hierarchy Visualization:")
    print("="*80)
    print("\nPattern Names by Level:")
    for level in ['node0', 'node1', 'node2', 'node3']:
        patterns = stats['patterns_by_level'][level]
        print(f"\n  {level.upper()} ({len(patterns)} patterns):")
        # Show first 3 patterns
        for i, pattern in enumerate(patterns[:3], 1):
            print(f"    {i}. {pattern}")
        if len(patterns) > 3:
            print(f"    ... and {len(patterns) - 3} more")

    # Step 6: Show metadata tracking
    print("\n[Step 6] Metadata Tracking:")
    print("="*80)
    print("\nMetadata automatically attached at each level:")
    print(f"  Book level: title='{book['title']}', author='{book['author']}'")
    print(f"  Chapter level: chapter_title, paragraph_count")
    print(f"  Paragraph level: paragraph_text, sentence_count")
    print(f"  Sentence level: sentence text")
    print("\nThis allows patterns to be traced back to their source!")

    print("\n" + "="*80)
    print("✓ Demonstration complete!")
    print("="*80)

    return learner, book

if __name__ == "__main__":
    learner, book = main()
    print("\n✓ Hierarchical concept learning demonstration successful!")
    print(f"\nTo query predictions from any node:")
    print(f"  node0_predictions = learner.nodes['node0'].get_predictions()")
    print(f"  node1_predictions = learner.nodes['node1'].get_predictions()")
    print(f"  etc.")
