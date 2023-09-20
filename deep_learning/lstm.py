import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self):
        super().__init__()

        mean, std = torch.tensor(0.0), torch.tensor(1.0)

        # LTM % to remember | fg = forget_gate | inp = input | hs = hidden_state
        self.fg_hs_w = nn.Parameter(
            torch.normal(mean=mean, std=std), requires_grad=True
        )
        self.fg_inp_w = nn.Parameter(
            torch.normal(mean=mean, std=std), requires_grad=True
        )
        self.fg_b = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        # Potential LTM % to remember | ig = input_gate | lb = left_block
        self.ig_lb_hs_w = nn.Parameter(
            torch.normal(mean=mean, std=std), requires_grad=True
        )
        self.ig_lb_in_w = nn.Parameter(
            torch.normal(mean=mean, std=std), requires_grad=True
        )
        self.ig_lb_b = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        # Potential LTM
        self.ig_rb_hs_w = nn.Parameter(
            torch.normal(mean=mean, std=std), requires_grad=True
        )
        self.ig_rb_in_w = nn.Parameter(
            torch.normal(mean=mean, std=std), requires_grad=True
        )
        self.ig_rb_b = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        # Potential STM
        # here we have only tanh activation function which will be implemented in forward method
        # and output of this finction will be multiplied by Potential LTM % from left_block and result is the updated STM.

        # Potential STM % to remember | og = output_gate
        self.og_lb_hs_w = nn.Parameter(
            torch.normal(mean=mean, std=std), requires_grad=True
        )
        self.og_lb_in_w = nn.Parameter(
            torch.normal(mean=mean, std=std), requires_grad=True
        )
        self.og_lb_b = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def lstm_unit(self, input_value, ltm, stm):
        # cell state component to restrict ltm to remember persentage
        ltm_to_remember_persentage = torch.sigmoid(
            (stm * self.fg_hs_w + input_value * self.fg_inp_w) + self.fg_b
        )

        # ptl = potential | ltm = long term memory | stm = sort term memory
        ptl_ltm_to_remember_persentage = torch.sigmoid(
            (stm * self.ig_lb_hs_w + input_value * self.ig_lb_in_w) + self.ig_lb_b
        )
        potential_ltm_to_remember = torch.tanh(
            (stm * self.ig_rb_hs_w + input_value * self.ig_rb_in_w) + self.ig_rb_b
        )

        # cell state component to update scaled ltm
        updated_ltm = (ltm * ltm_to_remember_persentage) + (
            ptl_ltm_to_remember_persentage * potential_ltm_to_remember
        )

        stm_to_remember_persentage = torch.sigmoid(
            (stm * self.og_lb_hs_w + input_value * self.og_lb_in_w) + self.og_lb_b
        )
        updated_stm = torch.tanh(updated_ltm) * stm_to_remember_persentage

        return [updated_ltm, updated_stm]

    def forward(self, input):
        ltm, stm = 0, 0

        day1 = input[0]
        day2 = input[1]
        day3 = input[2]
        day4 = input[3]

        ltm, stm = self.lstm_unit(day1, ltm, stm)
        ltm, stm = self.lstm_unit(day2, ltm, stm)
        ltm, stm = self.lstm_unit(day3, ltm, stm)
        ltm, stm = self.lstm_unit(day4, ltm, stm)

        return stm


if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = LSTM().to(device)
    print(model)
