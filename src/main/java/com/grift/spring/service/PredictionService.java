package com.grift.spring.service;

import com.grift.math.predictor.Predictor;
import com.grift.math.stats.ProbabilityVector;
import org.jetbrains.annotations.NotNull;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class PredictionService {

    private final Predictor predictor;

    @Autowired
    public PredictionService(Predictor predictor) {
        this.predictor = predictor;
    }

    @NotNull
    public ProbabilityVector makePrediction(@NotNull ProbabilityVector oldVec, @NotNull ProbabilityVector newVec) {
        return predictor.getPrediction(oldVec, newVec);
    }
}
