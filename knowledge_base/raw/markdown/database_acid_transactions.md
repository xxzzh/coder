# Database ACID Transactions

## Overview
ACID describes properties that help database transactions remain reliable despite errors and concurrency.

## Key Facts
- Atomicity means a transaction's operations are treated as an all-or-nothing unit.
- Consistency means a committed transaction moves the database from one valid state to another according to constraints.
- Isolation controls how concurrent transactions observe each other's intermediate or final effects.
- Durability means committed data should survive crashes or power loss, usually through logs, checkpoints, or replication.
- A transaction groups one or more operations into a logical unit of work.
- Indexes can speed up reads but add storage cost and can slow writes because index entries must be maintained.
- Normalization reduces avoidable duplication by organizing data into related tables with well-defined keys.

## Answerable Questions
- Q: What does ACID stand for?
  A: ACID stands for Atomicity, Consistency, Isolation, and Durability.
- Q: Why use indexes?
  A: Indexes speed up many queries, but they require storage and maintenance during writes.
