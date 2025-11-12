//
// Fragment shader for Gaussian splats.
// Renders a Gaussian splat within a quad, discarding fragments outside the unit circle.
// Applies an approximate Gaussian falloff based on distance from the center and outputs
// a color modulated by the alpha and Gaussian weight.
//

#ifdef OUTLINE_PASS
    // OUTLINE_PASS: Output white texture with alpha based on Gaussian function
    // Reference: PlayCanvas splat-shader.ts line 168-169
    void main() {
        float A = -dot(v_vertPos, v_vertPos);
        if (A < -4.0) {
            discard;
        }

        // Output white color with Gaussian falloff alpha
        float alpha = exp(A * 4.0) * v_splatColor.a;
        out_FragColor = vec4(1.0, 1.0, 1.0, alpha);
    }
#else
    // Normal rendering pass
    void main() {
        if (v_splitDirection < 0.0 && gl_FragCoord.x > czm_splitPosition) discard;
        if (v_splitDirection > 0.0 && gl_FragCoord.x < czm_splitPosition) discard;

        float A = -dot(v_vertPos, v_vertPos);
        if (A < -4.) {
            discard;
        }

        float B = exp(A * 4.) * v_splatColor.a ;
        out_FragColor = vec4(v_splatColor.rgb * B , B);
    }
#endif
