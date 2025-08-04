module SRT(
  input         clock,
                reset,
  output        input_ready,
  input         input_valid,
  input  [34:0] input_bits_dividend,
  input  [31:0] input_bits_divider,
  input  [4:0]  input_bits_counter,
  output        output_valid,
  output [31:0] output_bits_reminder,
                output_bits_quotient
);

  wire        input_ready_0;
  wire        input_valid_0 = input_valid;
  wire [34:0] input_bits_dividend_0 = input_bits_dividend;
  wire [31:0] input_bits_divider_0 = input_bits_divider;
  wire [4:0]  input_bits_counter_0 = input_bits_counter;
  SRT16 srt (
    .clock                (clock),
    .reset                (reset),
    .input_ready          (input_ready_0),
    .input_valid          (input_valid_0),
    .input_bits_dividend  (input_bits_dividend_0),
    .input_bits_divider   (input_bits_divider_0),
    .input_bits_counter   (input_bits_counter_0),
    .output_valid         (output_valid),
    .output_bits_reminder (output_bits_reminder),
    .output_bits_quotient (output_bits_quotient)
  );
  assign input_ready = input_ready_0;
endmodule

