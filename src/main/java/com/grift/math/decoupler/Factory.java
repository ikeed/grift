package com.grift.math.decoupler;

import lombok.NonNull;

public interface Factory {
    DecouplerMatrix make(@NonNull double[][] doubles);

    DecouplerMatrix make();
}
