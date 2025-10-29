"""
Storage layer for hierarchy metrics graph database.

Uses SQLite for persistence with schema optimized for graph queries.
"""

import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import time


@dataclass
class PatternNode:
    """Represents a single pattern in the hierarchy graph"""
    pattern_id: str              # Unique pattern identifier
    node_level: str              # e.g., 'node0', 'node1'
    frequency: int               # How many times pattern observed
    created_at: float            # Timestamp (epoch seconds)
    parent_ids: List[str]        # Patterns at level above that use this
    child_ids: List[str]         # Constituent patterns (level below)
    metadata: Optional[Dict] = None


@dataclass
class GraphCheckpoint:
    """Checkpoint snapshot during training"""
    checkpoint_id: int
    samples_processed: int
    timestamp: float
    pattern_counts: Dict[str, int]  # {node_level: count}
    metrics_snapshot: Dict[str, Any]


class HierarchyGraphStorage:
    """
    SQLite-based storage for hierarchy graph structure.

    Schema:
        patterns: Individual patterns with basic info
        edges: Parent-child relationships
        checkpoints: Training progress snapshots
        metadata: Configuration and run info
    """

    def __init__(self, db_path: str, verbose: bool = False):
        """
        Initialize storage.

        Args:
            db_path: Path to SQLite database file
            verbose: Print debug information
        """
        self.db_path = Path(db_path)
        self.verbose = verbose
        self._init_database()

    def _init_database(self):
        """Create database schema if not exists"""
        with self._connect() as conn:
            cursor = conn.cursor()

            # Patterns table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS patterns (
                    pattern_id TEXT PRIMARY KEY,
                    node_level TEXT NOT NULL,
                    frequency INTEGER DEFAULT 1,
                    created_at REAL NOT NULL,
                    metadata TEXT,

                    INDEX idx_level (node_level),
                    INDEX idx_created (created_at),
                    INDEX idx_frequency (frequency)
                )
            ''')

            # Edges table (parent-child relationships)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS edges (
                    parent_id TEXT NOT NULL,
                    child_id TEXT NOT NULL,
                    edge_type TEXT,
                    position INTEGER,
                    weight REAL DEFAULT 1.0,

                    PRIMARY KEY (parent_id, child_id),
                    FOREIGN KEY (parent_id) REFERENCES patterns(pattern_id),
                    FOREIGN KEY (child_id) REFERENCES patterns(pattern_id),

                    INDEX idx_parent (parent_id),
                    INDEX idx_child (child_id)
                )
            ''')

            # Checkpoints table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    samples_processed INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    pattern_counts TEXT NOT NULL,
                    metrics_snapshot TEXT,

                    INDEX idx_samples (samples_processed),
                    INDEX idx_timestamp (timestamp)
                )
            ''')

            # Metadata table (configuration, run info)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
            ''')

            conn.commit()

            if self.verbose:
                print(f"✓ Initialized database: {self.db_path}")

    @contextmanager
    def _connect(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()

    def add_pattern(
        self,
        pattern_id: str,
        node_level: str,
        frequency: int = 1,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Add or update a pattern.

        Args:
            pattern_id: Unique pattern identifier
            node_level: Hierarchy level (node0, node1, etc.)
            frequency: Pattern frequency
            metadata: Optional metadata dict
        """
        with self._connect() as conn:
            cursor = conn.cursor()

            metadata_json = json.dumps(metadata) if metadata else None

            cursor.execute('''
                INSERT OR REPLACE INTO patterns
                (pattern_id, node_level, frequency, created_at, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (pattern_id, node_level, frequency, time.time(), metadata_json))

            conn.commit()

    def add_patterns_batch(self, patterns: List[PatternNode]) -> None:
        """
        Batch insert patterns for performance.

        Args:
            patterns: List of PatternNode objects
        """
        with self._connect() as conn:
            cursor = conn.cursor()

            # Prepare batch data
            pattern_data = [
                (
                    p.pattern_id,
                    p.node_level,
                    p.frequency,
                    p.created_at,
                    json.dumps(p.metadata) if p.metadata else None
                )
                for p in patterns
            ]

            cursor.executemany('''
                INSERT OR REPLACE INTO patterns
                (pattern_id, node_level, frequency, created_at, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', pattern_data)

            conn.commit()

            if self.verbose:
                print(f"✓ Batch inserted {len(patterns)} patterns")

    def add_edge(
        self,
        parent_id: str,
        child_id: str,
        position: Optional[int] = None,
        weight: float = 1.0
    ) -> None:
        """
        Add parent-child relationship.

        Args:
            parent_id: Parent pattern ID
            child_id: Child pattern ID
            position: Position of child in parent sequence (0-indexed)
            weight: Edge weight (default 1.0)
        """
        with self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR IGNORE INTO edges
                (parent_id, child_id, position, weight)
                VALUES (?, ?, ?, ?)
            ''', (parent_id, child_id, position, weight))

            conn.commit()

    def add_edges_batch(self, edges: List[Tuple[str, str, Optional[int], float]]) -> None:
        """
        Batch insert edges for performance.

        Args:
            edges: List of (parent_id, child_id, position, weight) tuples
        """
        with self._connect() as conn:
            cursor = conn.cursor()

            cursor.executemany('''
                INSERT OR IGNORE INTO edges
                (parent_id, child_id, position, weight)
                VALUES (?, ?, ?, ?)
            ''', edges)

            conn.commit()

            if self.verbose:
                print(f"✓ Batch inserted {len(edges)} edges")

    def get_pattern(self, pattern_id: str) -> Optional[Dict]:
        """Get pattern by ID"""
        with self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT * FROM patterns WHERE pattern_id = ?
            ''', (pattern_id,))

            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def get_patterns_by_level(self, node_level: str) -> List[Dict]:
        """Get all patterns at a specific level"""
        with self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT * FROM patterns WHERE node_level = ?
                ORDER BY frequency DESC
            ''', (node_level,))

            return [dict(row) for row in cursor.fetchall()]

    def get_pattern_count_by_level(self) -> Dict[str, int]:
        """Get pattern counts for each level"""
        with self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT node_level, COUNT(*) as count
                FROM patterns
                GROUP BY node_level
                ORDER BY node_level
            ''')

            return {row['node_level']: row['count'] for row in cursor.fetchall()}

    def get_parents(self, pattern_id: str) -> List[str]:
        """Get all parent pattern IDs for a pattern"""
        with self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT parent_id FROM edges WHERE child_id = ?
            ''', (pattern_id,))

            return [row['parent_id'] for row in cursor.fetchall()]

    def get_children(self, pattern_id: str, ordered: bool = True) -> List[str]:
        """
        Get all child pattern IDs for a pattern.

        Args:
            pattern_id: Parent pattern ID
            ordered: Return in sequence order (by position)
        """
        with self._connect() as conn:
            cursor = conn.cursor()

            order_clause = 'ORDER BY position' if ordered else ''
            cursor.execute(f'''
                SELECT child_id FROM edges
                WHERE parent_id = ?
                {order_clause}
            ''', (pattern_id,))

            return [row['child_id'] for row in cursor.fetchall()]

    def get_parent_count_distribution(self, node_level: str) -> Dict[int, int]:
        """
        Get distribution of parent counts for patterns at a level.

        Returns:
            Dict mapping parent_count → number of patterns with that count
            E.g., {0: 100, 1: 50, 2: 30} means 100 orphans, 50 with 1 parent, etc.
        """
        with self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT
                    COALESCE(parent_counts.count, 0) as parent_count,
                    COUNT(*) as pattern_count
                FROM patterns p
                LEFT JOIN (
                    SELECT child_id, COUNT(*) as count
                    FROM edges
                    GROUP BY child_id
                ) parent_counts ON p.pattern_id = parent_counts.child_id
                WHERE p.node_level = ?
                GROUP BY parent_count
                ORDER BY parent_count
            ''', (node_level,))

            return {row['parent_count']: row['pattern_count'] for row in cursor.fetchall()}

    def get_orphan_patterns(self, node_level: str) -> List[str]:
        """Get patterns with no parents"""
        with self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT p.pattern_id
                FROM patterns p
                LEFT JOIN edges e ON p.pattern_id = e.child_id
                WHERE p.node_level = ? AND e.parent_id IS NULL
            ''', (node_level,))

            return [row['pattern_id'] for row in cursor.fetchall()]

    def get_coverage(self, child_level: str, parent_level: str) -> float:
        """
        Get coverage: % of child_level patterns used in parent_level.

        Args:
            child_level: Lower level (e.g., 'node0')
            parent_level: Upper level (e.g., 'node1')

        Returns:
            Coverage ratio (0.0-1.0)
        """
        with self._connect() as conn:
            cursor = conn.cursor()

            # Count total patterns at child level
            cursor.execute('''
                SELECT COUNT(*) as total FROM patterns WHERE node_level = ?
            ''', (child_level,))
            total = cursor.fetchone()['total']

            if total == 0:
                return 0.0

            # Count child patterns that appear in edges with parent_level parents
            cursor.execute('''
                SELECT COUNT(DISTINCT e.child_id) as used
                FROM edges e
                JOIN patterns p_parent ON e.parent_id = p_parent.pattern_id
                JOIN patterns p_child ON e.child_id = p_child.pattern_id
                WHERE p_parent.node_level = ? AND p_child.node_level = ?
            ''', (parent_level, child_level))
            used = cursor.fetchone()['used']

            return used / total

    def add_checkpoint(
        self,
        samples_processed: int,
        pattern_counts: Dict[str, int],
        metrics_snapshot: Optional[Dict] = None
    ) -> int:
        """
        Add training checkpoint.

        Returns:
            checkpoint_id
        """
        with self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO checkpoints
                (samples_processed, timestamp, pattern_counts, metrics_snapshot)
                VALUES (?, ?, ?, ?)
            ''', (
                samples_processed,
                time.time(),
                json.dumps(pattern_counts),
                json.dumps(metrics_snapshot) if metrics_snapshot else None
            ))

            conn.commit()
            return cursor.lastrowid

    def get_checkpoints(self) -> List[GraphCheckpoint]:
        """Get all checkpoints ordered by samples_processed"""
        with self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT * FROM checkpoints ORDER BY samples_processed
            ''')

            checkpoints = []
            for row in cursor.fetchall():
                checkpoints.append(GraphCheckpoint(
                    checkpoint_id=row['checkpoint_id'],
                    samples_processed=row['samples_processed'],
                    timestamp=row['timestamp'],
                    pattern_counts=json.loads(row['pattern_counts']),
                    metrics_snapshot=json.loads(row['metrics_snapshot']) if row['metrics_snapshot'] else {}
                ))

            return checkpoints

    def set_metadata(self, key: str, value: Any) -> None:
        """Store metadata (config, run info, etc.)"""
        with self._connect() as conn:
            cursor = conn.cursor()

            value_json = json.dumps(value) if not isinstance(value, str) else value

            cursor.execute('''
                INSERT OR REPLACE INTO metadata (key, value, updated_at)
                VALUES (?, ?, ?)
            ''', (key, value_json, time.time()))

            conn.commit()

    def get_metadata(self, key: str) -> Optional[Any]:
        """Retrieve metadata"""
        with self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT value FROM metadata WHERE key = ?
            ''', (key,))

            row = cursor.fetchone()
            if row:
                try:
                    return json.loads(row['value'])
                except json.JSONDecodeError:
                    return row['value']
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        with self._connect() as conn:
            cursor = conn.cursor()

            # Total patterns
            cursor.execute('SELECT COUNT(*) as total FROM patterns')
            total_patterns = cursor.fetchone()['total']

            # Total edges
            cursor.execute('SELECT COUNT(*) as total FROM edges')
            total_edges = cursor.fetchone()['total']

            # Patterns by level
            pattern_counts = self.get_pattern_count_by_level()

            # Checkpoints
            cursor.execute('SELECT COUNT(*) as total FROM checkpoints')
            total_checkpoints = cursor.fetchone()['total']

            # Database size
            db_size_mb = self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0

            return {
                'total_patterns': total_patterns,
                'total_edges': total_edges,
                'pattern_counts_by_level': pattern_counts,
                'total_checkpoints': total_checkpoints,
                'database_size_mb': db_size_mb,
            }

    def close(self):
        """Cleanup (no-op for SQLite, connections auto-close)"""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
