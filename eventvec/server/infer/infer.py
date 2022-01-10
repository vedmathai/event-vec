def sample(category, start_letter='A'):
    with torch.no_grad():
        category_tensor = categoryTensor(category).cuda()
        input = inputTensor(start_letter).cuda()
        hidden = rnn.initHidden().cuda()

        output_name = start_letter

        for i in range(max_length):
            output, hidden = rnn(category_tensor, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters -1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter).cuda()
    return output_name
