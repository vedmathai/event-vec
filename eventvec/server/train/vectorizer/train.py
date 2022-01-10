from torch import optim


def train_step(category_tensor, input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1).cuda()
    hidden = rnn.initHidden().cuda()

    rnn.zero_grad()
    optimizer = optim.SGD(rnn.parameters(), lr=learning_rate)

    loss = 0
    for i in range(input_line_tensor.size(0)):
        print(category_tensor.shape, input_line_tensor[i].shape)
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l

    loss.backward()
    optimizer.step()
    return output, loss.item() / input_line_tensor.size(0)


start = time.time()

def train():
    total_loss = 0
    all_losses = []
    for iter in range(1, n_iters + 1):
        category_tensor, input_line_tensor, target_line_tensor = randomTrainingExample()
        category_tensor, input_line_tensor, target_line_tensor = category_tensor.cuda(), input_line_tensor.cuda(), target_line_tensor.cuda()
        output, loss = train(category_tensor, input_line_tensor, target_line_tensor)
        total_loss += loss

        if iter % print_every == 0:
            print('%s (%d %d%%) %.4f' %(timeSince(start), iter, iter/n_iters*100, loss))

            all_losses.append(total_loss/plot_every)
            total_loss = 0
            if iter % (plot_every * print_every) == 0:
                plt.figure()
                plt.plot(all_losses)
                plt.show()


    max_length = 20