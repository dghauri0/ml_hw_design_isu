module MAC_8bit(ifmap, weights, lastdata, accumulation, en, clk, reset);

input [7:0] ifmap, weights;

input en, clk, reset, lastdata;

output reg [7:0] accumulation;

reg [7:0] temp;

always @(posedge clk) 
begin
    if(reset) begin
        accumulation <= 0; 
        temp         <= 0;
    end else if (en) begin
        if(lastdata) begin
            accumulation <= (ifmap*weights)+temp;
	    temp         <= 0;
        end else
            temp <= (ifmap*weights)+temp;
    end
end

endmodule
