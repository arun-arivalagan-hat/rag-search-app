# Database Indexing

An index is an auxiliary data structure that lets a database find rows without scanning the whole table. Without an index, a query such as `WHERE email = 'a@example.com'` must read every row, which is O(n). With a suitable index, lookup time drops to O(log n).

## B-tree indexes

The default index type in most relational databases, including PostgreSQL and MySQL, is the B-tree. It keeps keys in sorted order across a shallow, wide tree, so a lookup touches only a handful of disk pages even for tables with hundreds of millions of rows. B-trees support equality, range queries and ORDER BY on the indexed column.

## Hash and other indexes

A hash index maps a key to a bucket and gives fast equality lookups but cannot answer range queries. Specialised indexes also exist: GIN indexes for full-text search and arrays, and GiST indexes for geometric data.

## Trade-offs

Indexes speed up reads but cost something on writes: every INSERT, UPDATE or DELETE must also update each index, and indexes consume disk space. A composite index on (last_name, first_name) can serve queries filtering on last_name alone, but not on first_name alone. Use EXPLAIN to see whether the planner chooses an index scan or a sequential scan, and index the columns that appear in frequent filters and joins.
