//? mpirun -n 3

//! This examples demonstrates the use of the ghost communicator.
//!
//! We have three processes. The first process has indices 0, 1, 2, 3, 4.
//! The second process has indices 5, 6, 7, 8, 9. The third process has
//! indices 10, 11, 12, 13, 14.
//! The first process requires indices 5 and 6 from the second process as
//! ghost indices.
//! The second process requires index 10 from the third process as ghost index.
//! The third process requires index 5 from the second process and indices
//! 0, 1, 2 from the first process as ghosts.
//!
//! The example finally builds a second communicator in which the owners
//! choose a different number of values for each index.

#[cfg(not(feature = "mpi"))]
fn main() {
    println!("WARNING: MPI not enabled.");
}

#[cfg(feature = "mpi")]
pub fn main() {
    use mpi::traits::Communicator;
    use rlst::distributed_tools::{ChunkSizes, GhostCommunicatorBuilder};

    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();

    // This example is designed for 3 MPI ranks.

    if world.size() != 3 {
        println!("Please run this example with 3 MPI ranks.");
        return;
    }

    // We setup the ghost communicator. Each process adds the ghosts it
    // requires, together with their owning ranks.

    let ghosts: &[(usize, usize)] = if rank == 0 {
        &[(5, 1), (6, 1)]
    } else if rank == 1 {
        &[(10, 2)]
    } else {
        &[(5, 1), (0, 0), (1, 0), (2, 0)]
    };

    let ghost_comm = GhostCommunicatorBuilder::new()
        .add_ghosts(ghosts.iter().copied())
        .build(&world);

    // The communicator decides the order in which ghost values arrive: grouped
    // by owning rank, and in insertion order for the same owner. The third
    // process added index 5 first, but receives it last.

    if rank == 2 {
        assert_eq!(ghost_comm.receive_indices(), [0, 1, 2, 5]);
    }

    // Let us now send some data over the ghost communicator.

    let data = if rank == 0 {
        vec![10, 11, 12]
    } else if rank == 1 {
        vec![13, 14, 13]
    } else {
        vec![15]
    };

    let mut received_data = vec![0; ghost_comm.total_receive_count()];

    ghost_comm.forward_send_values(&data, &mut received_data);

    if rank == 0 {
        assert_eq!(received_data[0], 13);
        assert_eq!(received_data[1], 14);
    } else if rank == 1 {
        assert_eq!(received_data[0], 15);
    } else {
        assert_eq!(received_data[0], 10);
        assert_eq!(received_data[1], 11);
        assert_eq!(received_data[2], 12);
        assert_eq!(received_data[3], 13);
    }

    // Let us print the in-ranks, out-ranks, send and receive counts, the
    // send and receive indices, and the sent and received data of every
    // process as a table. Each rank sends its row to rank 0, which prints the
    // rows in rank order so that output from different processes does not
    // interleave.

    {
        use mpi::traits::{Destination, Source};

        let row = [
            rank.to_string(),
            format!("{:?}", ghost_comm.in_ranks()),
            format!("{:?}", ghost_comm.out_ranks()),
            format!("{:?}", ghost_comm.send_counts()),
            format!("{:?}", ghost_comm.receive_counts()),
            format!("{:?}", ghost_comm.send_indices()),
            format!("{:?}", ghost_comm.receive_indices()),
            format!("{:?}", data),
            format!("{:?}", received_data),
        ]
        .join("\t");

        if rank == 0 {
            let mut rows = vec![row];
            for other in 1..world.size() {
                let (bytes, _) = world.process_at_rank(other).receive_vec::<u8>();
                rows.push(String::from_utf8(bytes).unwrap());
            }

            let header = [
                "Rank",
                "In ranks",
                "Out ranks",
                "Send counts",
                "Receive counts",
                "Send indices",
                "Receive indices",
                "Sent data",
                "Received data",
            ];
            let rows: Vec<Vec<String>> = rows
                .iter()
                .map(|row| row.split('\t').map(String::from).collect())
                .collect();
            let widths: Vec<usize> = (0..header.len())
                .map(|col| {
                    rows.iter()
                        .map(|row| row[col].len())
                        .chain([header[col].len()])
                        .max()
                        .unwrap()
                })
                .collect();

            let format_row = |cells: &[&str]| {
                cells
                    .iter()
                    .zip(&widths)
                    .map(|(cell, width)| format!("{cell:<width$}"))
                    .collect::<Vec<_>>()
                    .join(" | ")
            };

            println!("{}", format_row(&header));
            println!(
                "{}",
                widths
                    .iter()
                    .map(|width| "-".repeat(*width))
                    .collect::<Vec<_>>()
                    .join("-+-")
            );
            for row in &rows {
                let cells: Vec<&str> = row.iter().map(String::as_str).collect();
                println!("{}", format_row(&cells));
            }
        } else {
            world.process_at_rank(0).send(row.as_bytes());
        }
    }

    // We now want to send the received data back to the original owners.

    let mut send_data = vec![0; ghost_comm.total_send_count()];

    ghost_comm.backward_send_values(&received_data, &mut send_data);

    if rank == 0 {
        assert_eq!(send_data[0], 10);
        assert_eq!(send_data[1], 11);
        assert_eq!(send_data[2], 12);
    } else if rank == 1 {
        assert_eq!(send_data[0], 13);
        assert_eq!(send_data[1], 14);
        assert_eq!(send_data[2], 13);
    } else {
        assert_eq!(send_data[0], 15);
    }

    // Finally, the owners choose the number of values for each of their
    // indices: index `i` carries `i % 3 + 1` values. The chunk sizes are
    // fixed when the communicator is built and reach the receivers there.

    let chunked_comm = GhostCommunicatorBuilder::new()
        .add_ghosts(ghosts.iter().copied())
        .chunk_sizes(ChunkSizes::PerIndex(Box::new(|index: usize| index % 3 + 1)))
        .build(&world);

    // The owners fill their send buffer. The values of the `i`-th send index
    // are at `send_offsets()[i]..send_offsets()[i + 1]`.

    let offsets = chunked_comm.send_offsets();
    let mut chunked_data = vec![0; chunked_comm.send_buffer_len()];
    for (i, &index) in chunked_comm.send_indices().iter().enumerate() {
        for (k, value) in chunked_data[offsets[i]..offsets[i + 1]]
            .iter_mut()
            .enumerate()
        {
            *value = 100 * index + k;
        }
    }

    let mut chunked_received = vec![0; chunked_comm.receive_buffer_len()];
    chunked_comm.forward_send_values(&chunked_data, &mut chunked_received);

    let offsets = chunked_comm.receive_offsets();
    for (i, &index) in chunked_comm.receive_indices().iter().enumerate() {
        let chunk = &chunked_received[offsets[i]..offsets[i + 1]];
        assert_eq!(chunk.len(), index % 3 + 1);
        for (k, &value) in chunk.iter().enumerate() {
            assert_eq!(value, 100 * index + k);
        }
    }

    if rank == 2 {
        println!(
            "Rank 2 received indices {:?} with offsets {:?}: {:?}",
            chunked_comm.receive_indices(),
            chunked_comm.receive_offsets(),
            chunked_received
        );
    }

    // Sending the chunks back reproduces the original send buffer.

    let mut chunked_returned = vec![0; chunked_comm.send_buffer_len()];
    chunked_comm.backward_send_values(&chunked_received, &mut chunked_returned);
    assert_eq!(chunked_returned, chunked_data);
}
