// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Controlled reproduction for <https://github.com/apache/datafusion/issues/25035>.
//!
//! The benchmark changes one mechanism-relevant variable at a time:
//! conjunction association, prefix selectivity, input batch width, number of
//! conjuncts, and the cost of the final conjunct. All generated inputs are
//! deterministic, and every left/right pair is checked for equal output before
//! it is measured.

use std::{hint::black_box, sync::Arc, time::Duration};

use arrow::{
    array::{ArrayRef, Int32Array, Int64Array, StringArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use datafusion_expr::{ColumnarValue, Operator};
use datafusion_physical_expr::{
    PhysicalExpr,
    expressions::{BinaryExpr, Column, lit},
};

const ROWS: usize = 8192;

#[derive(Debug, Clone, Copy)]
enum Suffix {
    Cheap,
    Regex,
}

#[derive(Debug, Clone, Copy)]
struct Case {
    conjuncts: usize,
    pass_permille: u16,
    payload_columns: usize,
    suffix: Suffix,
}

impl Case {
    fn label(self) -> String {
        let suffix = match self.suffix {
            Suffix::Cheap => "cheap",
            Suffix::Regex => "regex",
        };
        format!(
            "k{}_p{}_w{}_{}",
            self.conjuncts, self.pass_permille, self.payload_columns, suffix
        )
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

fn make_batch(case: Case) -> RecordBatch {
    assert!(case.conjuncts >= 2);
    assert!(case.pass_permille <= 1000);

    let cheap_conjuncts = match case.suffix {
        Suffix::Cheap => case.conjuncts,
        Suffix::Regex => case.conjuncts - 1,
    };
    let mut fields = Vec::with_capacity(
        cheap_conjuncts
            + usize::from(matches!(case.suffix, Suffix::Regex))
            + case.payload_columns,
    );
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(fields.capacity());

    for column in 0..cheap_conjuncts {
        let values = (0..ROWS)
            .map(|row| {
                let hash = splitmix64((row as u64) ^ ((column as u64 + 1) << 32));
                if hash % 1000 < u64::from(case.pass_permille) {
                    1
                } else {
                    -1
                }
            })
            .collect::<Vec<i32>>();
        fields.push(Field::new(
            format!("predicate_{column}"),
            DataType::Int32,
            false,
        ));
        arrays.push(Arc::new(Int32Array::from(values)));
    }

    if matches!(case.suffix, Suffix::Regex) {
        let values = (0..ROWS)
            .map(|row| {
                if row % 2 == 0 {
                    "https://api.example.com/v2/items/12345?source=benchmark"
                } else {
                    "http://unrelated.invalid/plain-text"
                }
            })
            .collect::<Vec<_>>();
        fields.push(Field::new("regex_input", DataType::Utf8, false));
        arrays.push(Arc::new(StringArray::from(values)));
    }

    for column in 0..case.payload_columns {
        let values = (0..ROWS)
            .map(|row| splitmix64((row as u64) ^ ((column as u64 + 101) << 32)) as i64)
            .collect::<Vec<_>>();
        fields.push(Field::new(
            format!("payload_{column}"),
            DataType::Int64,
            false,
        ));
        arrays.push(Arc::new(Int64Array::from(values)));
    }

    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap()
}

fn leaves(case: Case) -> Vec<Arc<dyn PhysicalExpr>> {
    let cheap_conjuncts = match case.suffix {
        Suffix::Cheap => case.conjuncts,
        Suffix::Regex => case.conjuncts - 1,
    };
    let mut leaves = (0..cheap_conjuncts)
        .map(|column| {
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new(&format!("predicate_{column}"), column)),
                Operator::Gt,
                lit(0_i32),
            )) as Arc<dyn PhysicalExpr>
        })
        .collect::<Vec<_>>();

    if matches!(case.suffix, Suffix::Regex) {
        leaves.push(Arc::new(BinaryExpr::new(
            Arc::new(Column::new("regex_input", cheap_conjuncts)),
            Operator::RegexMatch,
            lit(r"^https://([a-z]+\.)?example\.com/v[0-9]+/items/[0-9]+\?source=benchmark$"),
        )));
    }
    leaves
}

fn left_deep(mut leaves: Vec<Arc<dyn PhysicalExpr>>) -> Arc<dyn PhysicalExpr> {
    let mut expression = leaves.remove(0);
    for leaf in leaves {
        expression = Arc::new(BinaryExpr::new(expression, Operator::And, leaf));
    }
    expression
}

fn right_deep(mut leaves: Vec<Arc<dyn PhysicalExpr>>) -> Arc<dyn PhysicalExpr> {
    let mut expression = leaves.pop().unwrap();
    while let Some(leaf) = leaves.pop() {
        expression = Arc::new(BinaryExpr::new(leaf, Operator::And, expression));
    }
    expression
}

fn as_array(value: ColumnarValue) -> ArrayRef {
    value.into_array(ROWS).unwrap()
}

fn benchmark_issue_25035(c: &mut Criterion) {
    // The first case is the representative reproduction. The remainder are
    // controlled perturbations of one dimension around it.
    let cases = [
        Case {
            conjuncts: 8,
            pass_permille: 700,
            payload_columns: 32,
            suffix: Suffix::Cheap,
        },
        // Batch-width perturbation.
        Case {
            conjuncts: 8,
            pass_permille: 700,
            payload_columns: 0,
            suffix: Suffix::Cheap,
        },
        Case {
            conjuncts: 8,
            pass_permille: 700,
            payload_columns: 8,
            suffix: Suffix::Cheap,
        },
        Case {
            conjuncts: 8,
            pass_permille: 700,
            payload_columns: 64,
            suffix: Suffix::Cheap,
        },
        // Selectivity perturbation, including a no-preselection negative control.
        Case {
            conjuncts: 8,
            pass_permille: 500,
            payload_columns: 32,
            suffix: Suffix::Cheap,
        },
        Case {
            conjuncts: 8,
            pass_permille: 900,
            payload_columns: 32,
            suffix: Suffix::Cheap,
        },
        Case {
            conjuncts: 8,
            pass_permille: 990,
            payload_columns: 32,
            suffix: Suffix::Cheap,
        },
        // Conjunction-length perturbation.
        Case {
            conjuncts: 2,
            pass_permille: 700,
            payload_columns: 32,
            suffix: Suffix::Cheap,
        },
        Case {
            conjuncts: 4,
            pass_permille: 700,
            payload_columns: 32,
            suffix: Suffix::Cheap,
        },
        Case {
            conjuncts: 16,
            pass_permille: 700,
            payload_columns: 32,
            suffix: Suffix::Cheap,
        },
        // Positive control: preselection can be worthwhile with an expensive suffix.
        Case {
            conjuncts: 8,
            pass_permille: 700,
            payload_columns: 0,
            suffix: Suffix::Regex,
        },
    ];

    let mut group = c.benchmark_group("issue_25035");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(40);
    group.throughput(Throughput::Elements(ROWS as u64));

    for case in cases {
        let batch = make_batch(case);
        let left = left_deep(leaves(case));
        let right = right_deep(leaves(case));

        let left_result = as_array(left.evaluate(&batch).unwrap());
        let right_result = as_array(right.evaluate(&batch).unwrap());
        assert_eq!(
            left_result.as_ref(),
            right_result.as_ref(),
            "{}",
            case.label()
        );

        let label = case.label();
        group.bench_with_input(BenchmarkId::new("left", &label), &batch, |b, batch| {
            b.iter(|| black_box(left.evaluate(black_box(batch)).unwrap()))
        });
        group.bench_with_input(BenchmarkId::new("right", &label), &batch, |b, batch| {
            b.iter(|| black_box(right.evaluate(black_box(batch)).unwrap()))
        });
    }
    group.finish();
}

criterion_group!(benches, benchmark_issue_25035);
criterion_main!(benches);
