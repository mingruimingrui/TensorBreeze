from __future__ import print_function

import os

import tqdm
import time
import datetime

from collections import Mapping

from ..utils.weights_io import save_weights_to_file


def do_train(
    loss_dict,
    backprop_op,
    batch_size,
    max_iter,
    checkpoint_dir,
    checkpoint_period=10000,
    logging_period=200,
    sess=None,
    summary_writer=None,
    epoch_summaries=None,
    stdout=print,
):
    def save_checkpoint(checkpoint_dir, postfix):
        save_weights_to_file(os.path.join(
            checkpoint_dir,
            'model_{}.pkl'.format(postfix)
        ))

    # Check inputs are of correct type
    assert isinstance(loss_dict, Mapping)

    # Infer some variables
    should_update_tensorboard = summary_writer is not None \
        and epoch_summaries is not None
    loss_dict_ops = {k: v.op for k, v in loss_dict.items()}

    # Start training
    stdout('Start training')
    start_training_time = time.time()

    t0 = time.time()
    for iter_num in tqdm(range(1, max_iter + 1), ncols=60):
        batch_time = time.time() - t0
        t0 = time.time()

        if iter_num % logging_period != 0:
            sess.run([backprop_op, loss_dict_ops])

        else:
            # Update tensorboard if needed
            if should_update_tensorboard:
                epoch_summaries_buffer, losses, _ = \
                    sess.run([epoch_summaries, loss_dict, backprop_op])
                summary_writer.add_summary(epoch_summaries_buffer, iter_num)

            else:
                losses, _ = \
                    sess.run([loss_dict, backprop_op])

            # Compute eta
            time_elapsed = time.time() - start_training_time
            avg_batch_time = time_elapsed / iter_num
            eta_seconds = (max_iter - iter_num) * avg_batch_time
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            # Log training progress
            msg = {
                'eta': eta_string,
                'iter': iter_num,
                'batch_time': '{:.4f}s ({:.4f}s)'.format(batch_time, avg_batch_time),
                'losses': '{:.4f}'.format(losses)
            }
            stdout('{}'.format(msg))

        # Save state on checkpoint_period
        if iter_num % checkpoint_period == 0:
            save_checkpoint(checkpoint_dir, iter_num)

    # Save final model state and perform logging
    save_checkpoint(checkpoint_dir, 'final')
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))

    msg = 'Total training time: {} ({:.4f}s / it)'.format(
        total_time_str,
        total_training_time / max_iter
    )
    stdout(msg)
